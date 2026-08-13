"""
농업 문서 요약 및 질의응답 서비스
RAG + OpenAI Embeddings + FAISS + Streamlit
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv(r"C:\env\.env")

BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "faiss_index"

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150
EMBED_BATCH = 100

QA_SYSTEM_PROMPT = """당신은 농업 기술 문서를 근거로 답하는 전문가입니다.
제공된 컨텍스트(검색된 문서 조각)에 있는 내용만으로 답하세요.
컨텍스트에 없으면 추측하지 말고 "문서에서 확인되지 않습니다"라고 말하세요.
가능하면 출처(파일명, 페이지)를 언급하세요.
답변은 농가가 바로 참고할 수 있도록 명확한 한국어로 작성하세요."""

SUMMARY_SYSTEM_PROMPT = """당신은 농업 기술 문서를 요약하는 전문가입니다.
제공된 컨텍스트만 사용해 핵심을 정리하세요.
없는 내용은 보태지 마세요.
출처(파일명, 페이지)를 함께 적으세요."""

COMPARE_SYSTEM_PROMPT = """당신은 농업 기술 문서를 비교 정리하는 전문가입니다.
두 작물(문서)의 컨텍스트만 사용해 항목별로 비교 요약을 작성하세요.
없는 항목은 "문서에서 확인되지 않습니다"라고 적으세요.
추측하지 마세요."""


def list_pdfs() -> list[Path]:
    files = [
        p
        for p in BASE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    ]
    return sorted(files, key=lambda p: p.name)


def extract_pdf_pages(pdf_path: Path, max_pages: int) -> list[tuple[int, str]]:
    reader = PdfReader(str(pdf_path))
    n_pages = min(len(reader.pages), max_pages)
    pages: list[tuple[int, str]] = []
    for i in range(n_pages):
        text = reader.pages[i].extract_text() or ""
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        if len(text) >= 30:
            pages.append((i + 1, text))
    return pages


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def load_and_chunk(
    pdf_paths: list[Path],
    max_pages: int,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for pdf_path in pdf_paths:
        pages = extract_pdf_pages(pdf_path, max_pages)
        for page_num, page_text in pages:
            for i, chunk in enumerate(split_text(page_text, chunk_size, overlap)):
                documents.append(
                    {
                        "page_content": chunk,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_num,
                            "chunk_id": i,
                        },
                    }
                )
    return documents


def get_client() -> OpenAI:
    return OpenAI()


def embed_texts(client: OpenAI, texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        batch = texts[i : i + EMBED_BATCH]
        resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
        ordered = sorted(resp.data, key=lambda d: d.index)
        vectors.extend([d.embedding for d in ordered])
    arr = np.array(vectors, dtype="float32")
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    arr = arr / np.clip(norms, 1e-12, None)
    return arr


def build_faiss_index(vectors: np.ndarray) -> faiss.Index:
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def index_signature(sources: list[str], max_pages: int) -> dict[str, Any]:
    return {
        "sources": sorted(sources),
        "max_pages": max_pages,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embed_model": EMBED_MODEL,
    }


def save_index(
    index: faiss.Index,
    documents: list[dict[str, Any]],
    signature: dict[str, Any],
) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Windows에서 한글 경로를 faiss.write_index가 처리하지 못하므로 바이트로 저장
    index_bytes = faiss.serialize_index(index)
    (INDEX_DIR / "index.faiss").write_bytes(
        np.asarray(index_bytes, dtype=np.uint8).tobytes()
    )
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(documents, ensure_ascii=False),
        encoding="utf-8",
    )
    (INDEX_DIR / "meta.json").write_text(
        json.dumps(signature, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_saved_index() -> tuple[faiss.Index, list[dict[str, Any]], dict[str, Any]] | None:
    index_path = INDEX_DIR / "index.faiss"
    chunks_path = INDEX_DIR / "chunks.json"
    meta_path = INDEX_DIR / "meta.json"
    if not (index_path.exists() and chunks_path.exists() and meta_path.exists()):
        return None
    raw = np.frombuffer(index_path.read_bytes(), dtype=np.uint8).copy()
    index = faiss.deserialize_index(raw)
    documents = json.loads(chunks_path.read_text(encoding="utf-8"))
    signature = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, documents, signature


def search(
    client: OpenAI,
    index: faiss.Index,
    documents: list[dict[str, Any]],
    query: str,
    top_k: int,
) -> list[dict[str, Any]]:
    if not documents or index.ntotal == 0:
        return []
    k = min(top_k, index.ntotal)
    query_vec = embed_texts(client, [query])
    scores, ids = index.search(query_vec, k)
    results: list[dict[str, Any]] = []
    for score, idx in zip(scores[0], ids[0]):
        if idx < 0:
            continue
        doc = documents[int(idx)]
        results.append(
            {
                "page_content": doc["page_content"],
                "metadata": doc["metadata"],
                "score": float(score),
            }
        )
    return results


def format_context(hits: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        blocks.append(
            f"[근거 {i}] 출처: {meta['source']} / p.{meta['page']}\n{hit['page_content']}"
        )
    return "\n\n".join(blocks)


def generate_answer(
    client: OpenAI,
    system_prompt: str,
    user_prompt: str,
) -> str:
    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def rag_answer(client: OpenAI, question: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "문서에서 확인되지 않습니다. 관련 청크를 찾지 못했습니다."
    context = format_context(hits)
    user_prompt = (
        f"아래는 농업 기술 문서에서 검색된 컨텍스트입니다.\n\n"
        f"{context}\n\n"
        f"질문: {question}\n\n"
        "컨텍스트에 근거해 답하세요. 없으면 '문서에서 확인되지 않습니다'라고 답하세요."
    )
    return generate_answer(client, QA_SYSTEM_PROMPT, user_prompt)


def summarize_hits(client: OpenAI, topic: str, hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "문서에서 확인되지 않습니다. 요약할 관련 청크가 없습니다."
    context = format_context(hits)
    user_prompt = (
        f"주제: {topic}\n\n"
        f"아래 컨텍스트만 사용해 핵심을 요약하세요.\n\n{context}"
    )
    return generate_answer(client, SUMMARY_SYSTEM_PROMPT, user_prompt)


def compare_crops(
    client: OpenAI,
    crop_a: str,
    crop_b: str,
    topics: list[str],
    hits_by_topic: dict[str, list[dict[str, Any]]],
) -> str:
    parts: list[str] = []
    for topic in topics:
        hits = hits_by_topic.get(topic, [])
        parts.append(f"## 항목: {topic}\n{format_context(hits) if hits else '(관련 청크 없음)'}")
    context = "\n\n".join(parts)
    user_prompt = (
        f"작물 A 문서: {crop_a}\n작물 B 문서: {crop_b}\n"
        f"비교 항목: {', '.join(topics)}\n\n"
        f"{context}\n\n"
        "각 항목을 표 또는 불릿으로 비교 요약하세요."
    )
    return generate_answer(client, COMPARE_SYSTEM_PROMPT, user_prompt)


def render_hits(hits: list[dict[str, Any]]) -> None:
    if not hits:
        st.info("검색된 근거 청크가 없습니다.")
        return
    st.subheader("근거 청크")
    for i, hit in enumerate(hits, start=1):
        meta = hit["metadata"]
        score = hit.get("score")
        score_txt = f" | 유사도 {score:.3f}" if isinstance(score, float) else ""
        with st.expander(
            f"{i}) {meta['source']} / p.{meta['page']}{score_txt}",
            expanded=(i == 1),
        ):
            st.write(hit["page_content"])


def init_state() -> None:
    defaults: dict[str, Any] = {
        "index": None,
        "documents": None,
        "signature": None,
        "index_ready": False,
        "status_msg": "",
        "qa_answer": "",
        "qa_hits": [],
        "sum_text": "",
        "sum_hits": [],
        "cmp_text": "",
        "cmp_hits": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_index(
    index: faiss.Index,
    documents: list[dict[str, Any]],
    signature: dict[str, Any],
    status_msg: str,
) -> None:
    st.session_state.index = index
    st.session_state.documents = documents
    st.session_state.signature = signature
    st.session_state.index_ready = True
    st.session_state.status_msg = status_msg


def require_index() -> bool:
    if not st.session_state.index_ready:
        st.warning("사이드바에서 PDF를 선택한 뒤 [인덱스 생성]을 먼저 실행하세요.")
        return False
    return True


def main() -> None:
    st.set_page_config(
        page_title="농업 문서 요약 및 질의응답 서비스",
        page_icon="🌾",
        layout="wide",
    )
    init_state()

    st.title("농업 문서 요약 및 질의응답 서비스")
    st.caption("농촌진흥청 농업기술길잡이 기반 RAG · FAISS 검색")

    pdfs = list_pdfs()
    pdf_names = [p.name for p in pdfs]
    default_names = [
        name
        for name in [
            "농업기술길잡이40_딸기.PDF",
            "농업기술길잡이28_고구마.PDF",
        ]
        if name in pdf_names
    ]
    if len(default_names) < 2:
        default_names = pdf_names[:2]

    with st.sidebar:
        st.header("인덱스 설정")
        selected_names = st.multiselect(
            "선택 문서",
            options=pdf_names,
            default=default_names,
        )
        max_pages = st.slider("최대 페이지 수", min_value=5, max_value=300, value=30, step=5)
        top_k = st.slider("Top-K", min_value=2, max_value=10, value=4)

        create_clicked = st.button("인덱스 생성", type="primary", use_container_width=True)
        load_clicked = st.button("저장된 인덱스 불러오기", use_container_width=True)

        if st.session_state.status_msg:
            st.success(st.session_state.status_msg)

        st.divider()
        st.markdown(
            """
**테스트 질문**
- 딸기 정식 시기와 주의점은?
- 고구마 저장 방법은?
- 사과 병해충 방제 방법을 알려줘
- 참외 재배에 적합한 토성은?
- 우주선은 어떻게 만드나요?
            """
        )

        with st.expander("확인 질문 답안"):
            st.markdown(
                """
1. **RAG 없이 LLM만 쓰면** 학습 데이터에 없는 재배 지침을 지어내거나, 품종·시기·약제를 혼동할 수 있다. 출처도 없다.
2. **chunk_size가 너무 크면** 한 조각에 주제가 섞여 검색 정밀도가 떨어지고 토큰이 낭비된다. **너무 작으면** 문맥이 끊겨 의미가 훼손된다. **overlap이 너무 크면** 중복이 늘고, **너무 작으면** 경계 문장이 유실된다.
3. FAISS는 **키워드 일치가 아니라 의미 유사도**(임베딩 벡터 거리/내적)로 검색한다.
4. 검색 청크를 프롬프트에 넣는 이유는 LLM이 **현재 문서의 사실**을 근거로 생성하도록 하기 위함이다.
5. 문서에 없으면 **"문서에서 확인되지 않습니다"**라고 답하고 추측을 금지하는 것이 안전하다.
                """
            )

    if create_clicked:
        if len(selected_names) < 2:
            st.sidebar.error("PDF를 2개 이상 선택하세요.")
        else:
            selected_paths = [BASE_DIR / name for name in selected_names]
            progress = st.sidebar.progress(0, text="PDF 로드 및 청킹 중...")
            try:
                documents = load_and_chunk(selected_paths, max_pages=max_pages)
                if not documents:
                    st.sidebar.error("추출된 텍스트가 없습니다. 다른 PDF나 페이지 수를 조정하세요.")
                else:
                    progress.progress(35, text=f"임베딩 중... (청크 {len(documents)}개)")
                    client = get_client()
                    vectors = embed_texts(
                        client, [d["page_content"] for d in documents]
                    )
                    progress.progress(80, text="FAISS 인덱스 저장 중...")
                    index = build_faiss_index(vectors)
                    signature = index_signature(selected_names, max_pages)
                    save_index(index, documents, signature)
                    set_index(
                        index,
                        documents,
                        signature,
                        f"인덱스 준비 완료 (청크 {len(documents)}개)",
                    )
                    progress.progress(100, text="완료")
            except Exception as e:  # noqa: BLE001
                st.sidebar.error(f"인덱스 생성 오류: {e}")
            finally:
                progress.empty()

    if load_clicked:
        saved = load_saved_index()
        if saved is None:
            st.sidebar.error("저장된 인덱스가 없습니다. 먼저 인덱스를 생성하세요.")
        else:
            index, documents, signature = saved
            set_index(
                index,
                documents,
                signature,
                f"저장된 인덱스 로드 완료 (청크 {len(documents)}개)",
            )

    if not st.session_state.index_ready:
        saved = load_saved_index()
        if saved is not None and st.session_state.documents is None:
            index, documents, signature = saved
            set_index(
                index,
                documents,
                signature,
                f"기존 인덱스 자동 로드 (청크 {len(documents)}개)",
            )

    tab_qa, tab_sum, tab_cmp = st.tabs(["질의응답", "문서 요약", "작물 비교 요약"])

    with tab_qa:
        st.subheader("질의응답")
        question = st.text_input(
            "질문",
            placeholder="예: 딸기 정식 시기와 주의점은?",
        )
        if st.button("답변 생성", key="qa_btn"):
            if require_index() and question.strip():
                with st.spinner("관련 문서를 검색하고 답변을 생성하는 중..."):
                    try:
                        client = get_client()
                        hits = search(
                            client,
                            st.session_state.index,
                            st.session_state.documents,
                            question.strip(),
                            top_k,
                        )
                        st.session_state.qa_answer = rag_answer(
                            client, question.strip(), hits
                        )
                        st.session_state.qa_hits = hits
                    except Exception as e:  # noqa: BLE001
                        st.session_state.qa_answer = f"오류가 발생했습니다: {e}"
                        st.session_state.qa_hits = []
            elif not question.strip() and st.session_state.index_ready:
                st.warning("질문을 입력하세요.")

        if st.session_state.qa_answer:
            st.markdown("**답변**")
            st.write(st.session_state.qa_answer)
            render_hits(st.session_state.qa_hits)

    with tab_sum:
        st.subheader("문서 요약")
        mode = st.radio(
            "요약 방식",
            ["주제 요약", "선택 문서 요약"],
            horizontal=True,
        )
        if mode == "주제 요약":
            topic = st.text_input(
                "주제 또는 질문",
                placeholder="예: 토마토 육묘 관리",
                key="topic_input",
            )
            if st.button("주제 요약 생성", key="topic_btn"):
                if require_index() and topic.strip():
                    with st.spinner("관련 청크를 모아 요약하는 중..."):
                        try:
                            client = get_client()
                            hits = search(
                                client,
                                st.session_state.index,
                                st.session_state.documents,
                                topic.strip(),
                                top_k,
                            )
                            st.session_state.sum_text = summarize_hits(
                                client, topic.strip(), hits
                            )
                            st.session_state.sum_hits = hits
                        except Exception as e:  # noqa: BLE001
                            st.session_state.sum_text = f"오류가 발생했습니다: {e}"
                            st.session_state.sum_hits = []
                elif not topic.strip() and st.session_state.index_ready:
                    st.warning("주제를 입력하세요.")
        else:
            indexed_sources: list[str] = []
            if st.session_state.documents:
                indexed_sources = sorted(
                    {
                        d["metadata"]["source"]
                        for d in st.session_state.documents
                    }
                )
            doc_name = st.selectbox(
                "요약할 문서",
                options=indexed_sources or ["(인덱스가 없습니다)"],
            )
            if st.button("문서 요약 생성", key="doc_btn"):
                if require_index():
                    sample = [
                        {
                            "page_content": d["page_content"],
                            "metadata": d["metadata"],
                            "score": None,
                        }
                        for d in st.session_state.documents
                        if d["metadata"]["source"] == doc_name
                    ][:12]
                    with st.spinner("선택 문서 앞부분 청크를 요약하는 중..."):
                        try:
                            client = get_client()
                            st.session_state.sum_text = summarize_hits(
                                client, f"{doc_name} 핵심 내용", sample
                            )
                            st.session_state.sum_hits = sample
                        except Exception as e:  # noqa: BLE001
                            st.session_state.sum_text = f"오류가 발생했습니다: {e}"
                            st.session_state.sum_hits = []

        if st.session_state.sum_text:
            st.markdown("**요약**")
            st.write(st.session_state.sum_text)
            render_hits(st.session_state.sum_hits)

    with tab_cmp:
        st.subheader("멀티 문서 비교 요약")
        indexed_sources = []
        if st.session_state.documents:
            indexed_sources = sorted(
                {d["metadata"]["source"] for d in st.session_state.documents}
            )
        if len(indexed_sources) < 2:
            st.info("비교하려면 인덱스가 준비된 문서가 2개 이상 필요합니다.")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                crop_a = st.selectbox("문서 A", indexed_sources, index=0, key="cmp_a")
            with col_b:
                default_b = 1 if len(indexed_sources) > 1 else 0
                crop_b = st.selectbox(
                    "문서 B", indexed_sources, index=default_b, key="cmp_b"
                )
            topics = st.multiselect(
                "비교 항목",
                ["정식 조건", "병해충", "저장", "육묘", "토양"],
                default=["정식 조건", "병해충", "저장"],
            )
            if st.button("비교 요약 생성", key="cmp_btn"):
                if crop_a == crop_b:
                    st.warning("서로 다른 문서를 선택하세요.")
                elif not topics:
                    st.warning("비교 항목을 하나 이상 선택하세요.")
                else:
                    with st.spinner("항목별 관련 청크를 검색해 비교하는 중..."):
                        try:
                            client = get_client()
                            hits_by_topic: dict[str, list[dict[str, Any]]] = {}
                            all_hits: list[dict[str, Any]] = []
                            for topic in topics:
                                query = f"{crop_a}와 {crop_b}의 {topic}"
                                hits = search(
                                    client,
                                    st.session_state.index,
                                    st.session_state.documents,
                                    query,
                                    top_k,
                                )
                                hits_by_topic[topic] = hits
                                all_hits.extend(hits)
                            comparison = compare_crops(
                                client, crop_a, crop_b, topics, hits_by_topic
                            )
                        except Exception as e:  # noqa: BLE001
                            comparison = f"오류가 발생했습니다: {e}"
                            all_hits = []
                    st.session_state.cmp_text = comparison
                    st.session_state.cmp_hits = all_hits

            if st.session_state.cmp_text:
                st.markdown("**비교 요약**")
                st.write(st.session_state.cmp_text)
                render_hits(st.session_state.cmp_hits)


if __name__ == "__main__":
    main()
