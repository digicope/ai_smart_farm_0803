"""노트북 답안(RAG_검색전략_교체비교_실습.ipynb)과 동일한 RAG 검색 엔진.

Streamlit 예제와 노트북이 같은 벡터스토어·TOP_N·grounded 체인을 쓰도록 분리했다.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
import warnings
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.chains.query_constructor.schema import AttributeInfo
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers.self_query.base import SelfQueryRetriever
from langchain_community.query_constructors.chroma import ChromaTranslator
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

warnings.filterwarnings("ignore")

APP_DIR = Path(__file__).resolve().parent
ENV_PATH = Path(r"C:\env\.env")

EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
COLLECTION_NAME = "rda_annual_reports"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

GROUNDED_SYSTEM_PROMPT = """당신은 농촌진흥청 농촌진흥사업 연차보고서 기반 QA 어시스턴트입니다.
아래 [컨텍스트]에 포함된 내용만 근거로 답변하세요.
컨텍스트에 답이 없으면 "제공된 문서에서 답을 찾을 수 없습니다"라고 답하세요.
추측하거나 일반 지식을 보태지 마세요.
답변 마지막에는 참고한 출처를 (파일명, 연도, 페이지) 형식으로 표시하세요.

[컨텍스트]
{context}
"""

HYDE_SYSTEM_PROMPT = (
    "당신은 농촌진흥청 연차보고서 문체로 가상 문서를 작성하는 도우미입니다. "
    "주어진 질문에 대해, 실제 연차보고서에 있을 법한 답변 문단을 3~5문장으로 작성하세요. "
    "사실 여부를 과도하게 검증하지 말고, 검색에 쓸 '가상의 서술형 답변'만 본문으로 출력하세요."
)

SAMPLE_QUESTIONS = [
    "2024년 데이터 기반 스마트농업 확산과 고도화에서 어떤 성과가 있었나?",
    "가루쌀 품종 바로미2의 가공 이용 연구는 어떻게 이루어졌나?",
    "청년농업인 육성과 스마트 강소농 지원은 어떻게 추진되었나?",
]

MULTI_TURN_QUESTIONS = [
    "농촌진흥사업 기본계획은 무엇이며 왜 수립하는가?",
    "스마트농업 기술은 어떻게 확산·고도화되었나?",
    "2024년 아라온실 플랫폼의 성과는 무엇인가?",
    "탄소중립·환경친화적 농업기술에는 어떤 것이 있나?",
    "청년농업인 육성과 스마트 강소농 지원 내용은?",
    "앞서 설명한 스마트농업과 청년농업인 육성은 어떻게 연결되는가?",
    "KOPIA 등 국제협력 사업은 어떻게 추진되었나?",
]

STRATEGY_GUIDE = [
    {
        "situation": "질문이 짧고 문서 용어와 비슷함",
        "strategy": "Basic",
        "reason": "추가 LLM 호출 없이 가장 빠르고 저렴",
    },
    {
        "situation": "질문 표현이 보고서 문체와 다름 (추상·구어)",
        "strategy": "HyDE",
        "reason": "가상 답변 문서로 질문-문서 임베딩 간극을 줄임",
    },
    {
        "situation": "한 질문이 여러 하위 주제를 포함",
        "strategy": "MultiQuery",
        "reason": "재작성 쿼리 합집합으로 recall 향상. 비용·시간은 증가",
    },
    {
        "situation": '"2024년에는…"처럼 연도 조건이 명시됨',
        "strategy": "SelfQuery",
        "reason": "year 필터를 자동 추출해 다른 연도 노이즈를 배제",
    },
    {
        "situation": "후보를 넓게 모은 뒤 상위 정확도가 중요",
        "strategy": "Rerank",
        "reason": "BASE_K → Cross-Encoder/Cohere로 precision 향상",
    },
    {
        "situation": "대화가 길어지는 멀티턴",
        "strategy": "히스토리 압축",
        "reason": "오래된 턴 요약 + 최근 턴 원문으로 토큰을 줄이면서 맥락 유지",
    },
]


def load_api_keys() -> tuple[str, str | None]:
    load_dotenv(dotenv_path=ENV_PATH, override=True)
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY 가 C:\\env\\.env 에 없습니다.")
    return openai_key, os.getenv("COHERE_API_KEY") or None


def _pdf_candidates(root: Path) -> list[Path]:
    return [
        root / "pdf_data",
        root / "실습과제" / "01_RAG 검색 전략 교체 비교 실습" / "pdf_data",
        root / "수업중실습" / "pdf_data",
    ]


def resolve_pdf_dir() -> Path:
    """수업중실습/pdf_data 또는 같은 단원 실습과제 pdf_data 를 찾는다."""
    here = APP_DIR / "pdf_data"
    if here.is_dir() and list(here.glob("*.pdf")):
        return here

    search_roots = [APP_DIR, Path.cwd(), *APP_DIR.parents, *Path.cwd().parents]
    seen: set[str] = set()
    for root in search_roots:
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        for candidate in _pdf_candidates(root):
            if candidate.is_dir() and list(candidate.glob("*.pdf")):
                return candidate
    return here


def _normalize_hash_key(path: Path) -> str:
    """Windows에서 드라이브 대소문자만 달라도 MD5가 달라지므로 통일한다."""
    s = str(path)
    if len(s) >= 2 and s[1] == ":":
        s = s[0].lower() + s[1:]
    return s


def _chroma_usable(path: Path) -> bool:
    if not path.exists() or not (path / "chroma.sqlite3").exists():
        return False
    try:
        import chromadb

        client = chromadb.PersistentClient(path=str(path))
        return any(c.count() > 0 for c in client.list_collections())
    except Exception:
        return False


def resolve_persist_dir(base_dir: Path) -> Path:
    """Chroma/SQLite는 Windows에서 한글 경로를 열지 못하는 경우가 있다.

    노트북 답안과 같은 persist 폴더를 재사용한다. 경로 문자열이 달라도
    (C:\\ vs c:\\) 기존 인덱스가 있으면 그것을 우선한다.
    """
    local_dir = base_dir / "chroma_db"
    if str(local_dir).isascii() and _chroma_usable(local_dir):
        return local_dir

    digest = hashlib.md5(_normalize_hash_key(base_dir).encode("utf-8")).hexdigest()[:10]
    preferred = Path(tempfile.gettempdir()) / f"rda_rag_chroma_{digest}"
    if _chroma_usable(preferred):
        return preferred

    for d in sorted(Path(tempfile.gettempdir()).glob("rda_rag_chroma_*")):
        if _chroma_usable(d):
            return d

    if str(local_dir).isascii():
        return local_dir
    return preferred


def extract_year(filename: str) -> int:
    m = re.search(r"(20\d{2})", filename)
    if not m:
        raise ValueError(f"파일명에서 연도를 찾을 수 없습니다: {filename}")
    return int(m.group(1))


def _page_index(meta: dict) -> int:
    for key in ("page", "page_number"):
        if key in meta and meta[key] is not None:
            try:
                return int(meta[key])
            except (TypeError, ValueError):
                continue
    return -1


def _load_pdf_pages(path: str) -> list[Document]:
    try:
        from langchain_community.document_loaders import PyMuPDFLoader

        return PyMuPDFLoader(path).load()
    except Exception:
        from langchain_community.document_loaders import PyPDFLoader

        return PyPDFLoader(path).load()


def load_and_split_pdfs(pdf_dir: Path, chunk_size: int, chunk_overlap: int) -> list[Document]:
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))
    if len(pdf_paths) < 3:
        raise FileNotFoundError(
            f"pdf_data/ 에 연차보고서 PDF 3개가 필요합니다: {pdf_dir}\n"
            "수업중실습/pdf_data 또는 실습과제/01_RAG 검색 전략 교체 비교 실습/pdf_data 에 넣어 주세요."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    all_chunks: list[Document] = []
    for path in pdf_paths:
        filename = path.name
        year = extract_year(filename)
        pages = _load_pdf_pages(str(path))

        usable_pages = []
        for page in pages:
            text = (page.page_content or "").strip()
            if len(text) < 40:
                continue
            page.metadata["source"] = filename
            page.metadata["year"] = year
            page.metadata["page"] = _page_index(page.metadata)
            usable_pages.append(page)

        chunks = splitter.split_documents(usable_pages)
        for c in chunks:
            body = (c.page_content or "").strip()
            if len(body) < 80:
                continue
            c.metadata["source"] = filename
            c.metadata["year"] = year
            c.metadata["page"] = _page_index(c.metadata)
            all_chunks.append(c)
    return all_chunks


def format_docs(docs: list[Document]) -> str:
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "?")
        year = d.metadata.get("year", "?")
        page = d.metadata.get("page", "?")
        parts.append(f"[{i}] (source={src}, year={year}, page={page})\n{d.page_content}")
    return "\n\n".join(parts)


def doc_preview_rows(docs: list[Document]) -> list[dict]:
    rows = []
    for d in docs:
        snippet = " ".join((d.page_content or "").split())[:180]
        rows.append(
            {
                "source": d.metadata.get("source", "?"),
                "year": d.metadata.get("year", "?"),
                "page": d.metadata.get("page", "?"),
                "snippet": snippet,
            }
        )
    return rows


def _unique_union(docs: list[Document]) -> list[Document]:
    seen: set[tuple] = set()
    unique: list[Document] = []
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"), d.page_content[:120])
        if key in seen:
            continue
        seen.add(key)
        unique.append(d)
    return unique


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(LLM_MODEL)
        return len(encoding.encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 2)


class ConversationManager:
    """오래된 턴은 요약, 최근 KEEP_RECENT_TURNS 턴은 원문을 유지하는 대화 관리자."""

    def __init__(
        self,
        retrieve_fn,
        llm: ChatOpenAI,
        keep_recent_turns: int = 3,
        summarize_threshold: int = 6,
    ):
        self.retrieve_fn = retrieve_fn
        self.llm = llm
        self.keep_recent_turns = keep_recent_turns
        self.summarize_threshold = summarize_threshold
        self.turns: list[dict] = []
        self.summary: str = ""
        self.compression_events: list[dict] = []
        self._summarized_upto = 0

        self.summarize_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "다음은 기존 대화 요약과 새로 추가할 대화 턴입니다. "
                    "핵심 질문과 결론 위주로 3~6문장 이내의 새로운 누적 요약을 작성하세요.",
                ),
                ("human", "[기존 요약]\n{prev_summary}\n\n[추가할 턴]\n{new_turns}"),
            ]
        )
        self.summarize_chain = self.summarize_prompt | llm | StrOutputParser()

    def _build_history_context(self) -> str:
        parts = []
        if self.summary:
            parts.append(f"[이전 대화 요약]\n{self.summary}")
        recent = self.turns[-self.keep_recent_turns :]
        for t in recent:
            parts.append(f"Q: {t['question']}\nA: {t['answer']}")
        return "\n\n".join(parts)

    def _maybe_compress(self):
        if len(self.turns) <= self.summarize_threshold:
            return
        to_compress = self.turns[: -self.keep_recent_turns]
        new_chunk = to_compress[self._summarized_upto :]
        if not new_chunk:
            return
        new_turns_text = "\n\n".join(f"Q: {t['question']}\nA: {t['answer']}" for t in new_chunk)
        before_tokens = _count_tokens(self.summary)
        self.summary = self.summarize_chain.invoke(
            {
                "prev_summary": self.summary or "(없음)",
                "new_turns": new_turns_text,
            }
        )
        self._summarized_upto = len(to_compress)
        self.compression_events.append(
            {
                "compressed_turns": len(new_chunk),
                "summary_tokens_before": before_tokens,
                "summary_tokens_after": _count_tokens(self.summary),
            }
        )

    def ask(self, question: str) -> dict:
        history_context = self._build_history_context()
        docs, extras = self.retrieve_fn(question)
        doc_context = format_docs(docs)

        full_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GROUNDED_SYSTEM_PROMPT + "\n\n[이전 대화 히스토리]\n{history}"),
                ("human", "{question}"),
            ]
        )
        chain = full_prompt | self.llm | StrOutputParser()

        prompt_text = GROUNDED_SYSTEM_PROMPT.format(context=doc_context) + history_context + question
        history_tokens = _count_tokens(history_context)
        doc_tokens = _count_tokens(doc_context)
        prompt_tokens = _count_tokens(prompt_text)

        answer = chain.invoke(
            {"context": doc_context, "history": history_context, "question": question}
        )
        self.turns.append({"question": question, "answer": answer})
        compressed_before = len(self.compression_events)
        self._maybe_compress()

        return {
            "answer": answer,
            "docs": docs,
            "extras": extras,
            "prompt_tokens": prompt_tokens,
            "history_tokens": history_tokens,
            "doc_tokens": doc_tokens,
            "history_context": history_context,
            "just_compressed": len(self.compression_events) > compressed_before,
        }


class RAGEngine:
    """동일 벡터스토어 위에서 검색 전략만 교체한다."""

    def __init__(self, top_n: int = 4, base_k: int = 12, force_reindex: bool = False):
        self.top_n = top_n
        self.base_k = base_k
        self.openai_key, self.cohere_key = load_api_keys()
        self.pdf_dir = resolve_pdf_dir()
        self.persist_dir = resolve_persist_dir(APP_DIR)
        self.logs: list[str] = []

        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=self.openai_key)
        self.llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=self.openai_key)
        self.vectorstore = self._load_or_create_vectorstore(force_reindex=force_reindex)

        answer_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GROUNDED_SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        self.answer_chain = answer_prompt | self.llm | StrOutputParser()

        hyde_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", HYDE_SYSTEM_PROMPT),
                ("human", "질문: {question}"),
            ]
        )
        self.hyde_chain = hyde_prompt | self.llm | StrOutputParser()

        self.multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.top_n}),
            llm=self.llm,
            include_original=True,
        )

        metadata_field_info = [
            AttributeInfo(
                name="year",
                description="연차보고서가 다루는 연도. 2022, 2023, 2024 중 하나의 정수값.",
                type="integer",
            ),
            AttributeInfo(
                name="source",
                description="원본 PDF 파일명 (예: '2024년도_농촌진흥사업_연차보고서.pdf')",
                type="string",
            ),
            AttributeInfo(
                name="page",
                description="PDF 내 페이지 번호 (0-indexed)",
                type="integer",
            ),
        ]
        self.self_query_retriever = SelfQueryRetriever.from_llm(
            llm=self.llm,
            vectorstore=self.vectorstore,
            document_contents="농촌진흥청 농촌진흥사업 연차보고서의 본문 조각",
            metadata_field_info=metadata_field_info,
            structured_query_translator=ChromaTranslator(),
            search_kwargs={"k": self.top_n},
            enable_limit=True,
        )

        self._cross_encoder = None
        self._cohere_reranker = None
        if self.cohere_key:
            from langchain_cohere import CohereRerank

            self._cohere_reranker = CohereRerank(
                cohere_api_key=self.cohere_key,
                model="rerank-multilingual-v3.0",
                top_n=self.top_n,
            )

        self.doc_count = self.vectorstore._collection.count()
        self.logs.append(f"컬렉션 문서 수: {self.doc_count}")

    def _log(self, msg: str) -> None:
        self.logs.append(msg)

    def _load_or_create_vectorstore(self, force_reindex: bool) -> Chroma:
        collection_exists = _chroma_usable(self.persist_dir)
        if collection_exists and not force_reindex:
            self._log(f"기존 Chroma DB 로드: {self.persist_dir}")
            return Chroma(
                collection_name=COLLECTION_NAME,
                embedding_function=self.embeddings,
                persist_directory=str(self.persist_dir),
            )

        if force_reindex and self.persist_dir.exists():
            import shutil

            shutil.rmtree(self.persist_dir, ignore_errors=True)

        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._log(f"PDF 인덱싱 시작: {self.pdf_dir}")
        chunks = load_and_split_pdfs(self.pdf_dir, CHUNK_SIZE, CHUNK_OVERLAP)
        year_counts = dict(Counter(c.metadata["year"] for c in chunks))
        self._log(f"청크 {len(chunks)}개, 연도별 {year_counts}")
        t0 = time.time()
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=str(self.persist_dir),
        )
        self._log(f"인덱싱 완료: {time.time() - t0:.1f}s")
        return vs

    @property
    def strategy_names(self) -> list[str]:
        names = ["Basic", "HyDE", "MultiQuery", "SelfQuery", "Rerank(CrossEncoder)"]
        if self._cohere_reranker is not None:
            names.append("Rerank(Cohere)")
        return names

    def _get_cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            self._cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
        return self._cross_encoder

    def retrieve_basic(self, question: str, k: int | None = None) -> tuple[list[Document], dict]:
        k = k or self.top_n
        return self.vectorstore.similarity_search(question, k=k), {}

    def retrieve_hyde(self, question: str, k: int | None = None) -> tuple[list[Document], dict]:
        k = k or self.top_n
        hypo = self.hyde_chain.invoke({"question": question})
        docs = self.vectorstore.similarity_search(hypo, k=k)
        return docs, {"hypothetical_doc": hypo}

    def retrieve_multi_query(
        self, question: str, k: int | None = None, with_extras: bool = False
    ) -> tuple[list[Document], dict]:
        k = k or self.top_n
        extras: dict = {}
        if with_extras:
            try:
                if hasattr(self.multi_query_retriever, "generate_queries"):
                    extras["rewritten_queries"] = self.multi_query_retriever.generate_queries(question)
            except Exception as e:
                extras["rewritten_queries_error"] = str(e)
        docs = _unique_union(self.multi_query_retriever.invoke(question))
        return docs[:k], extras

    def retrieve_self_query(
        self, question: str, k: int | None = None, with_extras: bool = False
    ) -> tuple[list[Document], dict]:
        k = k or self.top_n
        extras: dict = {}
        docs = self.self_query_retriever.invoke(question)
        if with_extras:
            try:
                extras["structured_query"] = str(
                    self.self_query_retriever.query_constructor.invoke({"query": question})
                )
            except Exception as e:
                extras["structured_query"] = f"(파싱 실패) {e}"
        return docs[:k], extras

    def retrieve_rerank_cross_encoder(
        self, question: str, base_k: int | None = None, top_n: int | None = None
    ) -> tuple[list[Document], dict]:
        base_k = base_k or self.base_k
        top_n = top_n or self.top_n
        candidates = self.vectorstore.similarity_search(question, k=base_k)
        if not candidates:
            return [], {"base_k": 0, "scores": []}
        pairs = [(question, d.page_content) for d in candidates]
        scores = self._get_cross_encoder().predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
        top = reranked[:top_n]
        extras = {
            "base_k": len(candidates),
            "scores": [round(float(s), 4) for _, s in top],
        }
        return [d for d, _ in top], extras

    def retrieve_rerank_cohere(
        self, question: str, base_k: int | None = None, top_n: int | None = None
    ) -> tuple[list[Document], dict]:
        if self._cohere_reranker is None:
            raise RuntimeError("COHERE_API_KEY가 없어 Cohere Rerank를 사용할 수 없습니다.")
        base_k = base_k or self.base_k
        top_n = top_n or self.top_n
        candidates = self.vectorstore.similarity_search(question, k=base_k)
        if not candidates:
            return [], {"base_k": 0}
        reranked = self._cohere_reranker.compress_documents(documents=candidates, query=question)
        return list(reranked[:top_n]), {"base_k": len(candidates)}

    def retrieve(
        self, strategy: str, question: str, with_extras: bool = False
    ) -> tuple[list[Document], dict]:
        if strategy == "Basic":
            return self.retrieve_basic(question)
        if strategy == "HyDE":
            return self.retrieve_hyde(question)
        if strategy == "MultiQuery":
            return self.retrieve_multi_query(question, with_extras=with_extras)
        if strategy == "SelfQuery":
            return self.retrieve_self_query(question, with_extras=with_extras)
        if strategy == "Rerank(CrossEncoder)":
            return self.retrieve_rerank_cross_encoder(question)
        if strategy == "Rerank(Cohere)":
            return self.retrieve_rerank_cohere(question)
        raise ValueError(f"알 수 없는 전략: {strategy}")

    def generate_grounded_answer(self, question: str, docs: list[Document]) -> str:
        return self.answer_chain.invoke({"context": format_docs(docs), "question": question})

    def run_strategy(self, name: str, question: str, with_extras: bool = False) -> dict:
        t0 = time.time()
        docs, extras = self.retrieve(name, question, with_extras=with_extras)
        retrieval_time = time.time() - t0

        t0 = time.time()
        answer = self.generate_grounded_answer(question, docs)
        generation_time = time.time() - t0

        sources = [
            f"{d.metadata.get('source')}(y={d.metadata.get('year')}, p={d.metadata.get('page')})"
            for d in docs
        ]
        years_hit = sorted({d.metadata.get("year") for d in docs})

        return {
            "strategy": name,
            "question": question,
            "num_docs": len(docs),
            "years_hit": years_hit,
            "sources": sources,
            "retrieval_time_s": round(retrieval_time, 2),
            "generation_time_s": round(generation_time, 2),
            "total_time_s": round(retrieval_time + generation_time, 2),
            "answer": answer,
            "docs": docs,
            "extras": extras,
        }

    def make_conversation_manager(
        self,
        strategy: str = "Basic",
        keep_recent_turns: int = 3,
        summarize_threshold: int = 6,
    ) -> ConversationManager:
        return ConversationManager(
            retrieve_fn=lambda q: self.retrieve(strategy, q),
            llm=self.llm,
            keep_recent_turns=keep_recent_turns,
            summarize_threshold=summarize_threshold,
        )
