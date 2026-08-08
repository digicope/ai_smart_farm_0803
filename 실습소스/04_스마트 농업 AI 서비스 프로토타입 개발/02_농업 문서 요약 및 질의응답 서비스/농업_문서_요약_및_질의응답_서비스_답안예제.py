"""
농업 문서 요약 및 질의응답 서비스 답안 예제 (RAG + FAISS)

- 현재 폴더 PDF 문서 로드
- OpenAI Embeddings + FAISS 벡터 검색
- RAG 기반 질의응답 / 주제 요약
- Streamlit UI

실행:
    streamlit run 농업_문서_요약_및_질의응답_서비스_답안예제.py

또는 제출용으로 이름을 바꿔 실행:
    streamlit run app.py
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

# ---------------------------------------------------------------------------
# 환경 설정
# ---------------------------------------------------------------------------
load_dotenv()
load_dotenv("C:/env/.env")

API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

BASE_DIR = Path(__file__).resolve().parent


def resolve_index_dir() -> Path:
    """
    FAISS C++ IO는 Windows에서 한글 경로를 열지 못하는 경우가 많다.
    (Illegal byte sequence)

    프로젝트 경로에 비ASCII 문자가 있으면 TEMP 아래 ASCII 전용 폴더를 사용한다.
    """
    local_dir = BASE_DIR / "faiss_index"
    if str(local_dir).isascii():
        return local_dir

    digest = hashlib.md5(str(BASE_DIR).encode("utf-8")).hexdigest()[:10]
    return Path(tempfile.gettempdir()) / f"agri_rag_faiss_{digest}"


INDEX_DIR = resolve_index_dir()

SYSTEM_PROMPT = (
    "당신은 농업 기술 문서를 근거로 답하는 스마트 농업 전문 어시스턴트입니다. "
    "반드시 제공된 컨텍스트(문서 발췌) 내용만 사용해 답하세요. "
    "컨텍스트에 없으면 '제공된 문서에서 확인되지 않습니다'라고 명확히 말하세요. "
    "답변은 농민이 바로 활용할 수 있도록 간결하고 구체적으로 작성하세요. "
    "가능하면 어느 문서/페이지를 참고했는지 함께 언급하세요."
)


# ---------------------------------------------------------------------------
# PDF 유틸
# ---------------------------------------------------------------------------
def list_pdf_files() -> list[Path]:
    """현재 폴더의 PDF 파일 목록을 반환한다."""
    files = list(BASE_DIR.glob("*.pdf")) + list(BASE_DIR.glob("*.PDF"))
    # Windows에서 대소문자 중복 제거
    unique: dict[str, Path] = {}
    for f in files:
        unique[f.name.lower()] = f
    return sorted(unique.values(), key=lambda p: p.name)


def load_pdf_documents(pdf_path: Path, max_pages: int | None = None) -> list[Document]:
    """PDF에서 텍스트를 추출하고 Document 리스트로 반환한다."""
    loader = PyPDFLoader(str(pdf_path))
    pages = loader.load()

    if max_pages is not None and max_pages > 0:
        pages = pages[:max_pages]

    docs: list[Document] = []
    for page in pages:
        text = (page.page_content or "").strip()
        if not text:
            continue
        # 과도한 공백 정리
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        meta = dict(page.metadata or {})
        meta["source"] = pdf_path.name
        # langchain PyPDFLoader는 page를 0-index로 주는 경우가 많음
        page_no = meta.get("page", 0)
        try:
            meta["page"] = int(page_no) + 1
        except Exception:
            meta["page"] = page_no
        docs.append(Document(page_content=text, metadata=meta))
    return docs


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """문서를 검색용 청크로 분할한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(documents)


# ---------------------------------------------------------------------------
# FAISS 인덱스
# ---------------------------------------------------------------------------
def build_faiss_index(
    pdf_files: list[Path],
    max_pages: int,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[FAISS, int, int]:
    """
    선택한 PDF로 FAISS 인덱스를 구축한다.

    Returns:
        vectorstore, page_count, chunk_count
    """
    all_docs: list[Document] = []
    for pdf in pdf_files:
        all_docs.extend(load_pdf_documents(pdf, max_pages=max_pages))

    if not all_docs:
        raise ValueError("추출된 텍스트가 없습니다. PDF 선택/페이지 수를 확인하세요.")

    chunks = split_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        raise ValueError("청크가 생성되지 않았습니다. chunk_size 설정을 확인하세요.")

    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=API_KEY)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore, len(all_docs), len(chunks)


def save_faiss_index(vectorstore: FAISS) -> Path:
    """FAISS 인덱스를 ASCII 안전 경로에 저장하고 저장 경로를 반환한다."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Windows FAISS는 유니코드 경로에서 실패할 수 있어 resolve 후 문자열로 전달
    save_path = INDEX_DIR.resolve()
    vectorstore.save_local(str(save_path))
    return save_path


def load_faiss_index() -> FAISS | None:
    index_file = INDEX_DIR / "index.faiss"
    if not index_file.exists():
        return None
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL, api_key=API_KEY)
    return FAISS.load_local(
        str(INDEX_DIR.resolve()),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def search_with_scores(
    vectorstore: FAISS, query: str, k: int
) -> list[tuple[Document, float]]:
    """질문과 유사한 청크를 점수와 함께 검색한다."""
    return vectorstore.similarity_search_with_score(query, k=k)


def format_context(results: list[tuple[Document, float]]) -> str:
    """검색 결과를 LLM 컨텍스트 문자열로 만든다."""
    blocks = []
    for i, (doc, score) in enumerate(results, start=1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        blocks.append(
            f"[근거 {i}] 출처: {source} / p.{page} (거리점수: {score:.4f})\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks)


# ---------------------------------------------------------------------------
# LLM 생성 (요약 / 질의응답)
# ---------------------------------------------------------------------------
def ask_rag(client: OpenAI, question: str, context: str) -> str:
    """검색된 컨텍스트를 바탕으로 질의응답을 생성한다."""
    user_prompt = (
        "아래는 농업 기술 문서에서 검색한 근거입니다.\n\n"
        f"{context}\n\n"
        "위 근거만을 사용해 다음 질문에 답하세요.\n"
        f"질문: {question}\n\n"
        "답변 형식:\n"
        "1) 핵심 답변\n"
        "2) 실무 팁(있으면)\n"
        "3) 참고 출처"
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def summarize_topic(client: OpenAI, topic: str, context: str) -> str:
    """검색된 컨텍스트를 바탕으로 주제 요약을 생성한다."""
    user_prompt = (
        "아래는 농업 기술 문서에서 검색한 근거입니다.\n\n"
        f"{context}\n\n"
        f"주제: {topic}\n\n"
        "위 근거만을 사용해 3~6문장으로 핵심 요약을 작성하세요.\n"
        "불릿으로 정리하고, 마지막에 참고 출처를 적으세요."
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def summarize_document_sample(client: OpenAI, docs: list[Document], doc_name: str) -> str:
    """선택 문서의 앞부분 샘플 청크로 요약을 생성한다."""
    sample = docs[:12]
    context = "\n\n---\n\n".join(
        f"[p.{d.metadata.get('page', '?')}] {d.page_content[:900]}" for d in sample
    )
    user_prompt = (
        f"문서명: {doc_name}\n\n"
        "아래는 해당 문서의 일부 발췌입니다.\n\n"
        f"{context}\n\n"
        "문서의 전체 목차/핵심 주제를 추정해 요약하세요.\n"
        "- 문서 개요\n"
        "- 주요 재배 포인트 3~5개\n"
        "- 주의사항\n"
        "발췌에 없는 내용은 추측하지 마세요."
    )
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    return (response.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def init_session() -> None:
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "index_info" not in st.session_state:
        st.session_state.index_info = ""
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []


def render_evidence(results: list[tuple[Document, float]]) -> None:
    """검색 근거 청크를 접이식 UI로 표시한다."""
    with st.expander("검색된 근거 청크 보기", expanded=False):
        for i, (doc, score) in enumerate(results, start=1):
            source = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            st.markdown(f"**[{i}] {source} / p.{page}** · 거리점수 `{score:.4f}`")
            st.write(doc.page_content)
            st.divider()


def main() -> None:
    st.set_page_config(
        page_title="농업 문서 요약 및 질의응답 서비스",
        page_icon="🌿",
        layout="wide",
    )
    st.title("농업 문서 요약 및 질의응답 서비스")
    st.caption("RAG + FAISS · 농업기술길잡이 PDF 기반")

    init_session()

    if not API_KEY:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. `.env` 파일을 확인하세요.")
        st.stop()

    pdf_files = list_pdf_files()
    if not pdf_files:
        st.error("현재 폴더에 PDF 파일이 없습니다.")
        st.stop()

    client = OpenAI(api_key=API_KEY)

    with st.sidebar:
        st.subheader("1) 문서 선택")
        pdf_names = [p.name for p in pdf_files]
        selected_names = st.multiselect(
            "인덱싱할 PDF",
            options=pdf_names,
            default=pdf_names[:2],
        )

        st.subheader("2) 인덱싱 설정")
        max_pages = st.slider("최대 페이지 수(파일당)", 5, 80, 30, 5)
        chunk_size = st.slider("chunk_size", 400, 1200, 800, 50)
        chunk_overlap = st.slider("chunk_overlap", 50, 300, 150, 10)
        top_k = st.slider("Top-K 검색 개수", 2, 8, 4, 1)

        col_a, col_b = st.columns(2)
        with col_a:
            build_clicked = st.button("인덱스 생성", type="primary", use_container_width=True)
        with col_b:
            load_clicked = st.button("저장본 로드", use_container_width=True)

        save_after_build = st.checkbox("생성 후 로컬 저장", value=True)
        st.caption(f"인덱스 저장 경로\n`{INDEX_DIR}`")

        st.markdown("---")
        st.subheader("테스트 질문")
        st.markdown(
            """
- 딸기 정식 시기와 주의점은?
- 고구마 재배 환경은?
- 사과 병해충 방제 방법을 알려줘
- 참외 재배에 적합한 토성은?
- 우주선은 어떻게 만드나요?
            """
        )

        if st.button("대화 기록 초기화"):
            st.session_state.qa_history = []
            st.rerun()

    # 인덱스 생성
    if build_clicked:
        if not selected_names:
            st.sidebar.error("PDF를 하나 이상 선택하세요.")
        else:
            selected_paths = [BASE_DIR / name for name in selected_names]
            with st.spinner("PDF 로드 → 청킹 → 임베딩 → FAISS 구축 중..."):
                try:
                    vs, page_count, chunk_count = build_faiss_index(
                        selected_paths,
                        max_pages=max_pages,
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                    )
                    st.session_state.vectorstore = vs
                    st.session_state.index_info = (
                        f"준비 완료 · 문서 {len(selected_names)}개 · "
                        f"페이지 {page_count}개 · 청크 {chunk_count}개"
                    )
                    if save_after_build:
                        try:
                            saved_path = save_faiss_index(vs)
                            st.session_state.index_info += f" · 저장: `{saved_path}`"
                        except Exception as save_exc:
                            # 메모리 인덱스는 유지하고 저장만 실패 처리
                            st.session_state.index_info += " · 저장 실패(메모리 인덱스만 사용)"
                            st.sidebar.warning(f"인덱스 저장 실패: {save_exc}")
                    st.sidebar.success(st.session_state.index_info)
                except Exception as exc:
                    st.sidebar.error(f"인덱스 생성 실패: {exc}")

    # 저장본 로드
    if load_clicked:
        with st.spinner("저장된 FAISS 인덱스를 불러오는 중..."):
            try:
                vs = load_faiss_index()
                if vs is None:
                    st.sidebar.warning(
                        f"저장된 인덱스가 없습니다.\n경로: `{INDEX_DIR}`"
                    )
                else:
                    st.session_state.vectorstore = vs
                    st.session_state.index_info = f"저장본 로드 완료 · `{INDEX_DIR}`"
                    st.sidebar.success(st.session_state.index_info)
            except Exception as exc:
                st.sidebar.error(f"인덱스 로드 실패: {exc}")

    if st.session_state.index_info:
        st.info(st.session_state.index_info)
    else:
        st.warning("사이드바에서 PDF를 선택하고 **인덱스 생성**을 먼저 실행하세요.")

    tab_qa, tab_summary = st.tabs(["질의응답 (RAG)", "문서 요약"])

    # ------------------------------------------------------------------
    # 탭1: 질의응답
    # ------------------------------------------------------------------
    with tab_qa:
        st.subheader("문서 기반 질의응답")

        for item in st.session_state.qa_history:
            with st.chat_message("user"):
                st.markdown(item["question"])
            with st.chat_message("assistant"):
                st.markdown(item["answer"])

        question = st.chat_input("농업 문서에 대해 질문하세요")
        if question:
            if st.session_state.vectorstore is None:
                st.error("인덱스가 없습니다. 먼저 인덱스를 생성하거나 로드하세요.")
            else:
                with st.chat_message("user"):
                    st.markdown(question)

                with st.chat_message("assistant"):
                    with st.spinner("FAISS 검색 후 답변 생성 중..."):
                        try:
                            results = search_with_scores(
                                st.session_state.vectorstore, question, k=top_k
                            )
                            context = format_context(results)
                            answer = ask_rag(client, question, context)
                        except Exception as exc:
                            results = []
                            answer = f"오류가 발생했습니다: {exc}"

                    st.markdown(answer)
                    if results:
                        render_evidence(results)

                st.session_state.qa_history.append(
                    {"question": question, "answer": answer}
                )

    # ------------------------------------------------------------------
    # 탭2: 요약
    # ------------------------------------------------------------------
    with tab_summary:
        st.subheader("문서 요약")
        mode = st.radio(
            "요약 방식",
            ["주제 요약 (FAISS 검색 후 요약)", "선택 문서 샘플 요약"],
            horizontal=True,
        )

        if mode.startswith("주제"):
            topic = st.text_input(
                "요약할 주제",
                placeholder="예: 딸기 정식 시기와 주의사항",
            )
            if st.button("주제 요약 실행", type="primary"):
                if st.session_state.vectorstore is None:
                    st.error("인덱스가 없습니다. 먼저 인덱스를 생성하세요.")
                elif not topic.strip():
                    st.warning("주제를 입력하세요.")
                else:
                    with st.spinner("관련 청크 검색 후 요약 중..."):
                        try:
                            results = search_with_scores(
                                st.session_state.vectorstore, topic, k=top_k
                            )
                            context = format_context(results)
                            summary = summarize_topic(client, topic, context)
                            st.markdown(summary)
                            render_evidence(results)
                        except Exception as exc:
                            st.error(f"요약 실패: {exc}")
        else:
            target_name = st.selectbox("요약할 PDF", options=pdf_names)
            sample_pages = st.slider("앞에서부터 읽을 페이지 수", 3, 40, 15, 1)
            if st.button("문서 샘플 요약 실행", type="primary"):
                with st.spinner("문서 일부를 읽어 요약 중..."):
                    try:
                        docs = load_pdf_documents(
                            BASE_DIR / target_name, max_pages=sample_pages
                        )
                        chunks = split_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        summary = summarize_document_sample(client, chunks, target_name)
                        st.markdown(summary)
                        with st.expander("요약에 사용된 샘플 청크"):
                            for i, d in enumerate(chunks[:8], start=1):
                                st.markdown(
                                    f"**[{i}] p.{d.metadata.get('page', '?')}**"
                                )
                                st.write(d.page_content[:500])
                                st.divider()
                    except Exception as exc:
                        st.error(f"요약 실패: {exc}")


if __name__ == "__main__":
    main()
