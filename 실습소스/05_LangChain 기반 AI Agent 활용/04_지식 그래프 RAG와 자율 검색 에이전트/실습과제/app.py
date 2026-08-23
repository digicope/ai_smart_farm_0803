"""
농약 안전사용 상담 웹앱 (Streamlit)

노트북 `지식그래프RAG_자율검색에이전트_답안.ipynb`와 같은 자료·같은 전략을
브라우저에서 질문할 수 있게 옮긴 버전이다.

전략:
  - Naive Vector RAG
  - Graph RAG
  - Agentic RAG
  - Self-Corrective RAG (CRAG)
  - RAG Fusion
  - Hybrid (Dense + BM25)

실행 (이 폴더에서):
    streamlit run app.py

필요 패키지 (노트북과 동일 + streamlit):
    pip install streamlit rank_bm25 networkx beautifulsoup4 faiss-cpu langgraph
    pip install langchain-openai langchain-community langchain-classic pandas openpyxl python-dotenv lxml

API 키는 C:\\env\\.env 의 OPENAI_API_KEY 만 사용한다. 키 값은 화면에 출력하지 않는다.
"""

from __future__ import annotations

import operator
import os
import re
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Dict, List, Tuple, TypedDict

import networkx as nx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")

try:
    from langchain_classic.retrievers.ensemble import EnsembleRetriever
except Exception:
    from langchain.retrievers import EnsembleRetriever


# ---------------------------------------------------------------------------
# 설정
# ---------------------------------------------------------------------------
ENV_PATH = Path(r"C:\env\.env")
load_dotenv(dotenv_path=ENV_PATH, override=True)
os.environ.setdefault(
    "USER_AGENT",
    "Mozilla/5.0 (compatible; PesticideRAG-Lab/1.0; +https://psis.rda.go.kr)",
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

TARGET_CROPS = ["고추", "벼", "토마토", "사과", "배추"]
MAX_BRANDS_PER_CROP = 25
RANDOM_STATE = 42
TOP_N = 4
GRAPH_HOPS = 2
MAX_GRAPH_EDGES = 80
MAX_CRAG_RETRIES = 2

SAMPLE_QUESTIONS = [
    "고추 탄저병에 사용할 수 있는 저독성 제품의 회사와 안전사용시기는?",
    "벼에 쓰는 살충제를 만드는 회사는 어디인가?",
    "농약 중독 증상이 나타나면 어떻게 응급처치해야 하나요?",
    "농약 사용 후 빈 용기는 어떻게 관리해야 하나요?",
    "농약이란 무엇이며 신농약이 등록되기까지 왜 어려운가요?",
    "고충에 탄저 올때 저독성으로 뭐 뿌림?",
]

STRATEGIES = {
    "Naive Vector RAG": "제품 사실 문장을 질문과 비슷한 순으로 가져와 답한다.",
    "Graph RAG": "작물·병해충·제품·회사 관계를 k홉 순회해 멀티홉 질문에 답한다.",
    "Agentic RAG": "벡터/그래프/안전문서 도구를 Agent가 스스로 골라 호출한다.",
    "Self-Corrective RAG": "검색 결과를 채점하고, 부족하면 질의를 고치거나 웹 문서로 전환한다.",
    "RAG Fusion": "질문 변형을 여러 개 만든 뒤 RRF로 순위를 합친다.",
    "Hybrid (Dense+BM25)": "의미 검색과 키워드 검색 순위를 RRF로 섞는다.",
}


def resolve_assignment_dir() -> Path:
    """엑셀·URL 파일이 있는 실습과제 폴더를 찾는다."""
    markers = ["20260823_농약제품 목록.xlsx", "농약 관련 웹 문서 수집용 주소.txt"]
    here = Path(__file__).resolve().parent
    for p in [here, Path.cwd(), *Path.cwd().parents]:
        if all((p / m).exists() for m in markers):
            return p
        inner = p / "실습과제"
        if all((inner / m).exists() for m in markers):
            return inner
    return here


ASSIGN_DIR = resolve_assignment_dir()
EXCEL_PATH = ASSIGN_DIR / "20260823_농약제품 목록.xlsx"
URL_FILE = ASSIGN_DIR / "농약 관련 웹 문서 수집용 주소.txt"

GROUNDED_SYSTEM = (
    "당신은 농약 안전사용 상담 어시스턴트다. 아래 [근거]에 있는 내용만으로 답하라. "
    "근거에 없는 제품·용법·독성은 추측하지 말고 '주어진 자료만으로는 알 수 없다'고 답하라. "
    "답변 끝에 근거 유형(제품목록 / 지식그래프 / 안전사용 웹문서)을 한 줄로 밝혀라.\n\n[근거]\n{context}"
)

AGENT_SYSTEM = """당신은 농약 안전사용 상담 어시스턴트다. 다음 검색 도구를 자유롭게 사용할 수 있다.

- vector_search: 제품 사실 문장 의미 검색. 단일 사실 조회, 정확한 상표명을 모를 때 실마리 찾기에 적합.
- graph_search: 개체명 기점 그래프 순회. 작물-병해충-제품-회사-독성을 여러 단계로 연결할 때 적합.
- safety_search: 농약 정의·중독·응급처치·주의사항·사용 후 관리 웹 문서 검색.

지침:
1. 검색 없이 답할 수 있는 질문(인사, 역할 소개)은 도구를 호출하지 말고 바로 답하라.
2. 어떤 도구가 적합한지, 몇 번 호출할지는 스스로 판단하라.
3. graph_search가 개체를 찾지 못하면 vector_search로 정확한 이름을 파악한 뒤 다시 graph_search를 시도하라.
4. 제품 선택과 안전 수칙이 함께 필요하면 도구를 조합하라.
5. 충분한 근거를 모았으면 검색을 멈추고 근거에만 기반해 답하라. 부족하면 모른다고 답하라.
6. 최종 답변에 근거 유형을 간단히 함께 제시하라."""


# ---------------------------------------------------------------------------
# 데이터 준비
# ---------------------------------------------------------------------------
def crop_group(name: str) -> str | None:
    n = str(name).strip()
    for crop in TARGET_CROPS:
        if n == crop or n.startswith(crop + "(") or n.startswith(crop + " "):
            return crop
        if crop == "토마토" and "토마토" in n:
            return crop
        if crop == "고추" and n.startswith("고추"):
            return crop
    return None


def cell(row, col) -> str:
    v = row.get(col, "")
    if pd.isna(v):
        return ""
    return str(v).strip()


def row_to_fact(row) -> str:
    brand = cell(row, "상표명")
    crop = cell(row, "작물명")
    pest = cell(row, "적용병해충")
    item = cell(row, "품목명")
    common = cell(row, "일반명")
    content = cell(row, "주성분함량")
    use = cell(row, "용도")
    tox = cell(row, "인축독성")
    fish = cell(row, "어독성")
    company = cell(row, "회사명")
    phi = cell(row, "안전사용시기")
    times = cell(row, "안전사용횟수")
    method = cell(row, "사용방법")
    timing = cell(row, "사용적기")
    form = cell(row, "제형")
    dilute = cell(row, "희석배수")
    amount = cell(row, "사용량")
    parts = [
        f"상표명 '{brand}'는 작물 '{crop}'의 '{pest}'에 사용하는 {use}제이다."
        if use
        else f"상표명 '{brand}'는 작물 '{crop}'의 '{pest}'에 사용한다."
    ]
    if item:
        parts.append(f"품목명은 {item}이다.")
    if common:
        extra = f"(함량 {content})" if content else ""
        parts.append(f"주성분(일반명)은 {common}{extra}이다.")
    if tox:
        parts.append(f"인축독성은 {tox}이다.")
    if fish:
        parts.append(f"어독성은 {fish}이다.")
    if company:
        parts.append(f"회사는 {company}이다.")
    if phi:
        parts.append(f"안전사용시기는 {phi}이다.")
    if times:
        parts.append(f"안전사용횟수는 {times}이다.")
    if method:
        parts.append(f"사용방법은 {method}이다.")
    if timing:
        parts.append(f"사용적기는 {timing}이다.")
    if form:
        parts.append(f"제형은 {form}이다.")
    if dilute and dilute != "-":
        parts.append(f"희석배수는 {dilute}이다.")
    if amount and amount != "-":
        parts.append(f"사용량은 {amount}이다.")
    return " ".join(parts)


def load_product_sample() -> pd.DataFrame:
    raw = pd.read_excel(EXCEL_PATH, header=2)
    df = raw.copy()
    df["작물군"] = df["작물명"].map(crop_group)
    df = df.dropna(subset=["작물군", "상표명"]).copy()
    df["상표명"] = df["상표명"].astype(str).str.strip()
    df = df[df["상표명"].ne("") & df["상표명"].ne("nan")]

    frames = []
    for crop in TARGET_CROPS:
        part = df[df["작물군"] == crop]
        brands = (
            part["상표명"]
            .drop_duplicates()
            .sample(n=min(MAX_BRANDS_PER_CROP, part["상표명"].nunique()), random_state=RANDOM_STATE)
        )
        frames.append(part[part["상표명"].isin(set(brands))])
    return pd.concat(frames, ignore_index=True)


def products_to_docs(products: pd.DataFrame) -> List[Document]:
    docs: List[Document] = []
    for _, row in products.iterrows():
        docs.append(
            Document(
                page_content=row_to_fact(row),
                metadata={
                    "source": "product_excel",
                    "doc_type": "product",
                    "brand": cell(row, "상표명"),
                    "crop": cell(row, "작물명"),
                    "crop_group": row["작물군"],
                    "pest": cell(row, "적용병해충"),
                    "company": cell(row, "회사명"),
                    "use": cell(row, "용도"),
                    "toxicity": cell(row, "인축독성"),
                },
            )
        )
    return docs


def parse_url_list(path: Path) -> List[Tuple[str, str]]:
    text = path.read_text(encoding="utf-8")
    items: List[Tuple[str, str]] = []
    last_title = "농약 안전사용정보"
    for line in text.splitlines():
        line = line.strip().lstrip("-").strip()
        if not line:
            continue
        m = re.search(r"https?://\S+", line)
        if m:
            items.append((last_title, m.group(0).rstrip(").,]")))
        elif re.search(r"[가-힣]", line) and "http" not in line and not line.startswith("["):
            last_title = re.sub(r"^[0-9]+\.\s*", "", line).strip(" :")
    return items


def load_web_chunks() -> List[Document]:
    import requests
    from bs4 import BeautifulSoup
    from langchain_community.document_loaders import WebBaseLoader

    raw_docs: List[Document] = []
    for title, url in parse_url_list(URL_FILE):
        page_text = ""
        try:
            loaded = WebBaseLoader(web_path=url).load()
            if loaded:
                page_text = loaded[0].page_content or ""
        except Exception:
            page_text = ""
        if len(page_text.strip()) < 200:
            try:
                resp = requests.get(
                    url,
                    headers={"User-Agent": os.environ.get("USER_AGENT", "Mozilla/5.0")},
                    timeout=30,
                )
                resp.raise_for_status()
                resp.encoding = resp.apparent_encoding or "utf-8"
                soup = BeautifulSoup(resp.text, "html.parser")
                for tag in soup(["script", "style", "nav", "header", "footer", "noscript"]):
                    tag.decompose()
                page_text = soup.get_text("\n", strip=True)
            except Exception:
                continue
        lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
        raw_docs.append(
            Document(
                page_content="\n".join(lines),
                metadata={"source": url, "title": title, "doc_type": "safety_web"},
            )
        )
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    return splitter.split_documents(raw_docs)


def format_docs(documents: List[Document]) -> str:
    return "\n".join(f"- {d.page_content}" for d in documents)


# ---------------------------------------------------------------------------
# 지식 그래프
# ---------------------------------------------------------------------------
def row_to_triples(row) -> List[Tuple[str, str, str]]:
    brand = cell(row, "상표명")
    triples: List[Tuple[str, str, str]] = []

    def add(rel, obj):
        obj = str(obj).strip() if obj is not None and not pd.isna(obj) else ""
        if brand and obj and obj not in {"-", "nan"}:
            triples.append((brand, rel, obj))

    add("USED_ON", cell(row, "작물명"))
    add("CONTROLS", cell(row, "적용병해충"))
    add("CONTAINS", cell(row, "일반명"))
    add("MANUFACTURED_BY", cell(row, "회사명"))
    add("HAS_TOXICITY", cell(row, "인축독성"))
    add("HAS_FISH_TOXICITY", cell(row, "어독성"))
    add("HAS_USE", cell(row, "용도"))
    add("HAS_PHI", cell(row, "안전사용시기"))
    add("HAS_FORMULATION", cell(row, "제형"))
    crop, pest = cell(row, "작물명"), cell(row, "적용병해충")
    if crop and pest:
        triples.append((crop, "AFFECTED_BY", pest))
    return triples


def build_graph(products: pd.DataFrame) -> nx.MultiDiGraph:
    seen = set()
    g = nx.MultiDiGraph()
    for _, row in products.iterrows():
        for t in row_to_triples(row):
            if t not in seen:
                seen.add(t)
                g.add_edge(t[0], t[2], relation=t[1])
    return g


def find_seed_entities(question: str, g: nx.MultiDiGraph) -> List[str]:
    seeds = [node for node in g.nodes() if node and str(node) in question]
    seeds.sort(key=len, reverse=True)
    return seeds


def k_hop_edges(g: nx.MultiDiGraph, seeds: List[str], k: int = 2) -> List[tuple]:
    visited = set(seeds)
    frontier = set(seeds)
    collected = set()
    for _ in range(k):
        nxt = set()
        for node in frontier:
            for _, neighbor, data in g.out_edges(node, data=True):
                collected.add((node, data["relation"], neighbor))
                if neighbor not in visited:
                    nxt.add(neighbor)
            for neighbor, _, data in g.in_edges(node, data=True):
                collected.add((neighbor, data["relation"], node))
                if neighbor not in visited:
                    nxt.add(neighbor)
        visited |= nxt
        frontier = nxt
        if not frontier:
            break
    return sorted(collected)


def edges_to_context(edges: List[tuple]) -> str:
    return "\n".join(f"- ({s}) -[{r}]-> ({o})" for s, r, o in edges)


def prioritize_edges(edges: List[tuple], seeds: List[str], limit: int = MAX_GRAPH_EDGES) -> List[tuple]:
    seed_set = set(seeds)

    def score(e):
        s, r, o = e
        return int(s in seed_set) + int(o in seed_set)

    return sorted(edges, key=score, reverse=True)[:limit]


def graph_retrieve(question: str, g: nx.MultiDiGraph, k: int = GRAPH_HOPS) -> dict:
    seeds = find_seed_entities(question, g)
    edges = k_hop_edges(g, seeds, k=k) if seeds else []
    edges = prioritize_edges(edges, seeds)
    return {"seeds": seeds, "edges": edges, "context": edges_to_context(edges)}


# ---------------------------------------------------------------------------
# 상태·스키마
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]


class CRAGState(TypedDict):
    question: str
    original_question: str
    documents: List[Document]
    retry_count: int
    used_web: bool
    generation: str
    trace: Annotated[List[str], operator.add]


class GradeDocument(BaseModel):
    binary_score: str = Field(description="문서가 질문에 답하는 데 관련이 있으면 'yes', 없으면 'no'")


class QueryVariations(BaseModel):
    queries: List[str] = Field(description="원래 질문을 서로 다른 표현·관점으로 재작성한 검색 질의 목록")


def reciprocal_rank_fusion(
    ranked_lists: List[List[Document]],
    k: int = 60,
    top_n: int = TOP_N,
) -> List[Tuple[Document, float]]:
    scores: Dict[str, float] = defaultdict(float)
    doc_by_key: Dict[str, Document] = {}
    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.page_content
            scores[key] += 1.0 / (k + rank)
            doc_by_key.setdefault(key, doc)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(doc_by_key[key], score) for key, score in fused[:top_n]]


# ---------------------------------------------------------------------------
# 파이프라인 (한 번만 구축해 캐시)
# ---------------------------------------------------------------------------
@dataclass
class Pipeline:
    products: pd.DataFrame
    product_docs: List[Document]
    web_chunks: List[Document]
    product_vs: FAISS
    web_vs: FAISS
    combined_vs: FAISS
    graph: nx.MultiDiGraph
    llm: ChatOpenAI
    answer_chain: object
    graph_answer_chain: object
    agentic_rag: object
    self_corrective_rag: object
    multi_query_chain: object
    bm25_retriever: BM25Retriever
    ensemble_retriever: object
    stats: dict = field(default_factory=dict)


def build_pipeline() -> Pipeline:
    """엑셀 샘플링 → 웹 수집 → FAISS·그래프·에이전트까지 한 번에 만든다."""
    if not OPENAI_API_KEY:
        raise RuntimeError(r"OPENAI_API_KEY 가 C:\env\.env 에 없습니다.")
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"엑셀 파일이 없습니다: {EXCEL_PATH}")
    if not URL_FILE.exists():
        raise FileNotFoundError(f"URL 파일이 없습니다: {URL_FILE}")

    llm = ChatOpenAI(model=LLM_MODEL, temperature=0, api_key=OPENAI_API_KEY)
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)

    products = load_product_sample()
    product_docs = products_to_docs(products)
    web_chunks = load_web_chunks()
    if not web_chunks:
        raise RuntimeError("안전사용 웹 문서를 수집하지 못했습니다. 네트워크를 확인하세요.")

    product_vs = FAISS.from_documents(product_docs, embeddings)
    web_vs = FAISS.from_documents(web_chunks, embeddings)
    combined_vs = FAISS.from_documents(product_docs + web_chunks, embeddings)
    graph = build_graph(products)

    answer_prompt = ChatPromptTemplate.from_messages(
        [("system", GROUNDED_SYSTEM), ("human", "{question}")]
    )
    answer_chain = answer_prompt | llm | StrOutputParser()

    graph_answer_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "아래 [관계] 목록은 농약 지식 그래프에서 질문과 관련해 순회로 수집한 사실이다. "
                "이 관계들을 연결해 질문에 답하라. 목록만으로 답할 수 없으면 "
                "'주어진 자료만으로는 알 수 없다'라고 답하라.\n\n[관계]\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    graph_answer_chain = graph_answer_prompt | llm | StrOutputParser()

    @tool
    def vector_search(query: str) -> str:
        """농약제품 사실 문장을 의미 유사도로 검색한다.
        'OO 상표의 주성분은?', '이 약의 회사는?'처럼 단일 사실 조회에 적합하다."""
        retrieved = product_vs.similarity_search(query, k=TOP_N)
        return format_docs(retrieved) if retrieved else "검색 결과 없음"

    @tool
    def graph_search(entity: str, hops: int = 2) -> str:
        """지식 그래프에서 entity를 시작점으로 최대 hops단계까지 관계를 순회한다.
        작물-병해충-제품-회사처럼 여러 단계를 연결해야 하는 질문에 적합하다."""
        seeds = [node for node in graph.nodes() if entity in str(node) or str(node) in entity]
        if not seeds:
            return f"'{entity}'와 일치하는 개체를 그래프에서 찾지 못함. 정확한 개체명 확인이 필요하다."
        edges = prioritize_edges(k_hop_edges(graph, seeds, k=hops), seeds)
        return edges_to_context(edges) if edges else f"'{entity}'에서 시작하는 관계를 찾지 못함"

    @tool
    def safety_search(query: str) -> str:
        """농약 정의, 중독 증상, 응급처치, 사용 시 주의사항, 사용 후 관리 웹 문서를 검색한다."""
        retrieved = web_vs.similarity_search(query, k=TOP_N)
        return format_docs(retrieved) if retrieved else "검색 결과 없음"

    tools = [vector_search, graph_search, safety_search]
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        messages = state["messages"]
        if not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=AGENT_SYSTEM)] + messages
        return {"messages": [llm_with_tools.invoke(messages)]}

    def should_continue(state: AgentState) -> str:
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return END

    workflow = StateGraph(AgentState)
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", ToolNode(tools))
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
    workflow.add_edge("tools", "agent")
    agentic_rag = workflow.compile()

    grade_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 검색된 문서가 사용자 질문과 관련이 있는지 채점하는 채점자다.\n"
                "문서에 질문과 관련된 키워드나 의미가 담겨 있으면 관련 있다고 판단한다.\n"
                "엄격한 정답 일치가 아니라, 답을 찾는 데 실제로 도움이 되는지를 기준으로 느슨하게 채점하되,\n"
                "명백히 무관한 주제라면 'no'로 채점하라.",
            ),
            ("human", "[검색된 문서]\n{document}\n\n[질문]\n{question}"),
        ]
    )
    grade_chain = grade_prompt | llm.with_structured_output(GradeDocument)

    rewrite_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "다음 질문으로 검색했지만 관련 문서를 찾지 못했다. 검색에 더 적합하도록 질문을 다시 작성하라.\n"
                "- 원래 질문의 의도는 그대로 유지한다.\n"
                "- 구어체·오탈자·축약 지칭을 작물명·병해충명·상표명 등 핵심어로 풀어 쓴다.\n"
                "- 재작성한 질문 한 줄만 출력하고, 다른 설명은 덧붙이지 않는다.",
            ),
            ("human", "원래 질문: {question}"),
        ]
    )
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    generate_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "아래 [근거] 문서만 사용해 질문에 답하라. 문서에 없는 내용은 추측하지 마라.\n"
                "답변 끝에 근거 출처가 제품목록(내부)인지 안전사용 웹문서(외부)인지 한 줄로 밝혀라.",
            ),
            ("human", "[근거 출처: {source}]\n{context}\n\n[질문]\n{question}"),
        ]
    )
    generate_chain = generate_prompt | llm | StrOutputParser()

    def retrieve(state: CRAGState) -> dict:
        documents = product_vs.similarity_search(state["question"], k=TOP_N)
        return {
            "documents": documents,
            "trace": [f"[검색] 질의 '{state['question']}' → 제품 KB {len(documents)}건"],
        }

    def grade_documents(state: CRAGState) -> dict:
        trace_lines = []
        relevant = []
        for d in state["documents"]:
            score = grade_chain.invoke(
                {"document": d.page_content, "question": state["original_question"]}
            )
            flag = "관련" if score.binary_score.lower() == "yes" else "무관"
            preview = d.page_content[:80].replace("\n", " ")
            trace_lines.append(f"    · [{flag}] {preview}")
            if score.binary_score.lower() == "yes":
                relevant.append(d)
        trace_lines.insert(0, f"[채점] 검색된 {len(state['documents'])}건 중 관련 문서 {len(relevant)}건")
        return {"documents": relevant, "trace": trace_lines}

    def transform_query(state: CRAGState) -> dict:
        rewritten = rewrite_chain.invoke({"question": state["question"]})
        return {
            "question": rewritten,
            "retry_count": state["retry_count"] + 1,
            "trace": [
                f"[재작성] '{state['question']}' → '{rewritten}' "
                f"(시도 {state['retry_count'] + 1}/{MAX_CRAG_RETRIES})"
            ],
        }

    def web_search(state: CRAGState) -> dict:
        documents = web_vs.similarity_search(state["original_question"], k=TOP_N)
        return {
            "documents": documents,
            "used_web": True,
            "trace": [f"[웹전환] 안전사용 웹문서에서 {len(documents)}건"],
        }

    def generate(state: CRAGState) -> dict:
        source = "안전사용 웹문서(외부)" if state.get("used_web") else "제품목록(내부)"
        if not state["documents"]:
            generation = "주어진 자료만으로는 알 수 없다.\n근거 출처: 없음"
            trace = ["[생성] 관련 근거 없음 → 모른다고 답변"]
        else:
            generation = generate_chain.invoke(
                {
                    "source": source,
                    "context": format_docs(state["documents"]),
                    "question": state["original_question"],
                }
            )
            trace = [f"[생성] {source} {len(state['documents'])}건을 근거로 최종 답변 작성"]
        return {"generation": generation, "trace": trace}

    def route_after_grade(state: CRAGState) -> str:
        if state["documents"]:
            return "generate"
        if state["retry_count"] < MAX_CRAG_RETRIES:
            return "transform_query"
        return "web_search"

    crag = StateGraph(CRAGState)
    crag.add_node("retrieve", retrieve)
    crag.add_node("grade_documents", grade_documents)
    crag.add_node("transform_query", transform_query)
    crag.add_node("web_search", web_search)
    crag.add_node("generate", generate)
    crag.set_entry_point("retrieve")
    crag.add_edge("retrieve", "grade_documents")
    crag.add_conditional_edges(
        "grade_documents",
        route_after_grade,
        {"generate": "generate", "transform_query": "transform_query", "web_search": "web_search"},
    )
    crag.add_edge("transform_query", "retrieve")
    crag.add_edge("web_search", "generate")
    crag.add_edge("generate", END)
    self_corrective_rag = crag.compile()

    multi_query_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 검색 질의를 다양화하는 도우미다. 아래 질문을 벡터 검색에 쓸 수 있도록 "
                "서로 다른 표현·핵심어·관점으로 재작성한 질의를 {n}개 만들어라.\n"
                "- 원래 질문의 의도는 유지하되, 동의어·상위 개념·세부 키워드 등 표현을 다양화한다.\n"
                "- 각 질의는 한 문장으로 작성하고, 질의끼리 서로 겹치지 않게 한다.",
            ),
            ("human", "{question}"),
        ]
    )
    multi_query_chain = multi_query_prompt | llm.with_structured_output(QueryVariations)

    bm25_retriever = BM25Retriever.from_documents(product_docs + web_chunks)
    bm25_retriever.k = TOP_N
    dense_retriever = combined_vs.as_retriever(search_kwargs={"k": TOP_N})
    ensemble_retriever = EnsembleRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )

    stats = {
        "product_rows": len(products),
        "brands": int(products["상표명"].nunique()),
        "product_docs": len(product_docs),
        "web_chunks": len(web_chunks),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "crops": (
            products.groupby("작물군")
            .agg(행수=("상표명", "size"), 상표수=("상표명", "nunique"))
            .reset_index()
        ),
    }

    return Pipeline(
        products=products,
        product_docs=product_docs,
        web_chunks=web_chunks,
        product_vs=product_vs,
        web_vs=web_vs,
        combined_vs=combined_vs,
        graph=graph,
        llm=llm,
        answer_chain=answer_chain,
        graph_answer_chain=graph_answer_chain,
        agentic_rag=agentic_rag,
        self_corrective_rag=self_corrective_rag,
        multi_query_chain=multi_query_chain,
        bm25_retriever=bm25_retriever,
        ensemble_retriever=ensemble_retriever,
        stats=stats,
    )


# ---------------------------------------------------------------------------
# 전략별 실행
# ---------------------------------------------------------------------------
def run_naive(pipe: Pipeline, question: str) -> dict:
    retrieved = pipe.product_vs.similarity_search(question, k=TOP_N)
    answer = pipe.answer_chain.invoke({"context": format_docs(retrieved), "question": question})
    return {"answer": answer, "docs": retrieved}


def run_graph(pipe: Pipeline, question: str) -> dict:
    retrieval = graph_retrieve(question, pipe.graph, k=GRAPH_HOPS)
    answer = pipe.graph_answer_chain.invoke(
        {"context": retrieval["context"], "question": question}
    )
    return {"answer": answer, **retrieval}


def run_agentic(pipe: Pipeline, question: str) -> dict:
    result = pipe.agentic_rag.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 15},
    )
    messages = result["messages"]
    steps: List[str] = []
    step = 1
    for m in messages[1:]:
        if getattr(m, "tool_calls", None):
            for call in m.tool_calls:
                steps.append(f"[{step}] 도구 호출: {call['name']}({call['args']})")
                step += 1
        elif getattr(m, "type", "") == "tool":
            preview = m.content if len(m.content) < 280 else m.content[:280] + " ..."
            steps.append(f"    └─ {preview}")
    answer = messages[-1].content if messages else ""
    return {"answer": answer, "trace": steps}


def run_crag(pipe: Pipeline, question: str) -> dict:
    init: CRAGState = {
        "question": question,
        "original_question": question,
        "documents": [],
        "retry_count": 0,
        "used_web": False,
        "generation": "",
        "trace": [],
    }
    result = pipe.self_corrective_rag.invoke(init, config={"recursion_limit": 15})
    return {"answer": result["generation"], "trace": result.get("trace", [])}


def run_fusion(pipe: Pipeline, question: str, n_variations: int = 4) -> dict:
    variations = pipe.multi_query_chain.invoke({"question": question, "n": n_variations}).queries
    queries = [question] + variations
    ranked = [pipe.combined_vs.similarity_search(q, k=TOP_N) for q in queries]
    fused = reciprocal_rank_fusion(ranked, top_n=TOP_N)
    docs = [d for d, _ in fused]
    answer = pipe.answer_chain.invoke({"context": format_docs(docs), "question": question})
    return {"answer": answer, "variations": queries, "fused": fused}


def run_hybrid(pipe: Pipeline, question: str) -> dict:
    dense_docs = pipe.combined_vs.similarity_search(question, k=TOP_N)
    sparse_docs = pipe.bm25_retriever.invoke(question)
    fused = reciprocal_rank_fusion([dense_docs, sparse_docs], top_n=TOP_N)
    docs = [d for d, _ in fused]
    answer = pipe.answer_chain.invoke({"context": format_docs(docs), "question": question})
    return {"answer": answer, "dense": dense_docs, "sparse": sparse_docs, "fused": fused}


def run_strategy(pipe: Pipeline, strategy: str, question: str) -> dict:
    t0 = time.time()
    if strategy == "Naive Vector RAG":
        out = run_naive(pipe, question)
    elif strategy == "Graph RAG":
        out = run_graph(pipe, question)
    elif strategy == "Agentic RAG":
        out = run_agentic(pipe, question)
    elif strategy == "Self-Corrective RAG":
        out = run_crag(pipe, question)
    elif strategy == "RAG Fusion":
        out = run_fusion(pipe, question)
    elif strategy == "Hybrid (Dense+BM25)":
        out = run_hybrid(pipe, question)
    else:
        raise ValueError(f"알 수 없는 전략: {strategy}")
    out["elapsed"] = time.time() - t0
    out["strategy"] = strategy
    return out


# ---------------------------------------------------------------------------
# UI 헬퍼
# ---------------------------------------------------------------------------
def render_payload(payload) -> None:
    """단일 결과 또는 전략 비교 결과 목록을 화면에 그린다."""
    if isinstance(payload, list):
        tabs = st.tabs([r.get("strategy", f"전략 {i}") for i, r in enumerate(payload, 1)])
        for tab, result in zip(tabs, payload):
            with tab:
                render_result(result)
        return
    render_result(payload)


def render_result(result: dict) -> None:
    st.markdown("#### 답변")
    st.write(result.get("answer", ""))
    st.caption(f"소요 {result.get('elapsed', 0):.1f}초 · {result.get('strategy', '')}")

    if result.get("docs"):
        with st.expander("검색된 제품 사실 문장", expanded=False):
            for i, d in enumerate(result["docs"], 1):
                meta = d.metadata
                st.markdown(
                    f"**[{i}]** {meta.get('brand', '')} / {meta.get('crop', '')} / {meta.get('pest', '')}"
                )
                st.write(d.page_content)
                st.divider()

    if "seeds" in result:
        with st.expander("그래프 순회 근거", expanded=True):
            st.write("시작 개체(seed):", ", ".join(result["seeds"][:12]) or "(없음)")
            st.write(f"수집 엣지 수: {len(result.get('edges', []))}")
            st.code(result.get("context") or "(관계 없음)", language="text")

    if result.get("trace"):
        with st.expander("실행 트레이스", expanded=True):
            st.text("\n".join(result["trace"]))

    if result.get("variations"):
        with st.expander("질의 변형", expanded=True):
            for i, q in enumerate(result["variations"]):
                label = "원본" if i == 0 else f"변형 {i}"
                st.write(f"**[{label}]** {q}")

    if result.get("fused"):
        with st.expander("RRF 융합 상위 문서", expanded=False):
            for i, item in enumerate(result["fused"], 1):
                doc, score = item
                src = (
                    doc.metadata.get("doc_type")
                    or doc.metadata.get("title")
                    or doc.metadata.get("source")
                )
                st.markdown(f"**{i}위** · score `{score:.4f}` · {src}")
                st.write(doc.page_content[:400])
                st.divider()

    if result.get("dense") is not None:
        cols = st.columns(2)
        with cols[0]:
            with st.expander("Dense(벡터) 결과"):
                for d in result["dense"]:
                    st.write("-", d.page_content[:120].replace("\n", " "))
        with cols[1]:
            with st.expander("BM25(키워드) 결과"):
                for d in result["sparse"]:
                    st.write("-", d.page_content[:120].replace("\n", " "))


# ---------------------------------------------------------------------------
# Streamlit 페이지
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="농약 안전사용 상담",
        page_icon="🌿",
        layout="wide",
    )
    st.title("🌿 농약 안전사용 상담")
    st.caption(
        "Graph RAG · Agentic RAG · Self-Corrective RAG · RAG Fusion  |  "
        "농약제품 목록 + 안전사용 웹문서"
    )

    if not OPENAI_API_KEY:
        st.error(r"OPENAI_API_KEY 가 C:\env\.env 에 없습니다. 키를 넣은 뒤 페이지를 새로고침하세요.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pipe" not in st.session_state:
        st.session_state.pipe = None

    with st.sidebar:
        st.subheader("검색 전략")
        strategy = st.radio(
            "사용할 전략",
            list(STRATEGIES.keys()),
            index=2,
        )
        st.info(STRATEGIES[strategy])

        compare = st.checkbox("같은 질문을 여러 전략으로 비교", value=False)
        compare_targets: List[str] = []
        if compare:
            compare_targets = st.multiselect(
                "비교할 전략",
                list(STRATEGIES.keys()),
                default=["Naive Vector RAG", "Graph RAG", "Agentic RAG"],
            )

        st.markdown("---")
        st.subheader("지식베이스")
        st.caption(f"엑셀: `{EXCEL_PATH.name}`")
        st.caption(f"URL: `{URL_FILE.name}`")
        st.caption("작물군: " + ", ".join(TARGET_CROPS) + f" · 작물별 상표 {MAX_BRANDS_PER_CROP}개")

        build_clicked = st.button("지식베이스 구축 / 새로고침", type="primary", use_container_width=True)
        if st.button("대화 기록 지우기", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        st.subheader("예시 질문")
        example = st.selectbox("골라 보내기", ["(직접 입력)"] + SAMPLE_QUESTIONS)
        if st.button("예시 질문 보내기", use_container_width=True):
            if example != "(직접 입력)":
                st.session_state.pending_question = example
                st.rerun()

    if build_clicked:
        with st.spinner(
            "엑셀 샘플링 · 웹 문서 수집 · 임베딩 · 그래프 구축 중... "
            "(최초 1회는 1~2분 걸릴 수 있습니다)"
        ):
            try:
                st.session_state.pipe = build_pipeline()
                st.session_state.messages = []
            except Exception as exc:
                st.session_state.pipe = None
                st.error(f"지식베이스 구축 실패: {exc}")
                st.stop()

    pipe = st.session_state.pipe
    if pipe is None:
        st.info("왼쪽에서 **지식베이스 구축 / 새로고침**을 눌러 제품 목록과 안전사용 문서를 인덱싱하세요.")
        st.stop()

    st.success("지식베이스가 준비되었습니다. 아래 채팅창에 질문하세요.")
    stats = pipe.stats
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("제품 행", f"{stats['product_rows']:,}")
    m2.metric("상표 수", f"{stats['brands']:,}")
    m3.metric("웹 청크", f"{stats['web_chunks']:,}")
    m4.metric("그래프 엣지", f"{stats['graph_edges']:,}")
    with st.expander("작물군별 샘플 현황"):
        st.dataframe(stats["crops"], use_container_width=True, hide_index=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.write(msg["content"])
            else:
                render_payload(msg["content"])

    pending = st.session_state.pop("pending_question", None)
    typed = st.chat_input("농약 제품·안전사용에 대해 질문하세요")
    question = pending or typed

    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            if compare and compare_targets:
                tabs = st.tabs(compare_targets)
                packed = []
                for tab, name in zip(tabs, compare_targets):
                    with tab:
                        with st.spinner(f"{name} 실행 중..."):
                            result = run_strategy(pipe, name, question)
                        render_result(result)
                        packed.append(result)
                st.session_state.messages.append({"role": "assistant", "content": packed})
            else:
                with st.spinner(f"{strategy} 실행 중..."):
                    result = run_strategy(pipe, strategy, question)
                render_result(result)
                st.session_state.messages.append({"role": "assistant", "content": result})
        except Exception as exc:
            st.error(f"실행 중 오류: {exc}")


if __name__ == "__main__":
    main()
