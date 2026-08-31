"""
RAG 검색 전략 교체 비교 — Streamlit 예제

노트북 답안(RAG_검색전략_교체비교_실습.ipynb)과 같은 엔진을 웹 UI로 실행한다.

실행:
    cd "실습과제/01_RAG 검색 전략 교체 비교 실습"
    streamlit run streamlit_rag_compare.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_engine import (
    EMBEDDING_MODEL,
    LLM_MODEL,
    MULTI_TURN_QUESTIONS,
    SAMPLE_QUESTIONS,
    STRATEGY_GUIDE,
    RAGEngine,
    doc_preview_rows,
)

st.set_page_config(
    page_title="RAG 검색 전략 비교",
    page_icon="🌾",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-title {
        font-size: 1.85rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
        color: #1b4332;
    }
    .subtitle {
        color: #52796f;
        font-size: 0.95rem;
        margin-bottom: 1.1rem;
    }
    .stMetric {
        background: #f7faf7;
        border: 1px solid #d8e3d8;
        border-radius: 10px;
        padding: 0.4rem 0.6rem;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #d8e3d8;
        border-radius: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def get_engine(top_n: int, base_k: int, force_reindex: bool) -> RAGEngine:
    return RAGEngine(top_n=top_n, base_k=base_k, force_reindex=force_reindex)


def _init_chat_manager(engine: RAGEngine, strategy: str, keep_recent: int, threshold: int):
    st.session_state.chat_mgr = engine.make_conversation_manager(
        strategy=strategy,
        keep_recent_turns=keep_recent,
        summarize_threshold=threshold,
    )
    st.session_state.chat_messages = []
    st.session_state.chat_metrics = []


def render_doc_table(docs) -> None:
    rows = doc_preview_rows(docs)
    if not rows:
        st.info("검색된 문서가 없습니다.")
        return
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render_extras(strategy: str, extras: dict) -> None:
    if not extras:
        return
    if strategy == "HyDE" and extras.get("hypothetical_doc"):
        with st.expander("HyDE 가상 문서 (검색용, 답변 근거 아님)", expanded=True):
            st.write(extras["hypothetical_doc"])
    if strategy == "MultiQuery":
        queries = extras.get("rewritten_queries")
        if queries:
            with st.expander("Multi-Query 재작성 질의", expanded=True):
                if isinstance(queries, str):
                    st.write(queries)
                else:
                    for i, q in enumerate(queries, 1):
                        st.markdown(f"{i}. {q}")
    if strategy == "SelfQuery" and extras.get("structured_query"):
        with st.expander("Self-Query 구조화 쿼리", expanded=True):
            st.code(str(extras["structured_query"]))
    if strategy.startswith("Rerank"):
        cols = st.columns(2)
        cols[0].metric("1차 후보(BASE_K)", extras.get("base_k", "-"))
        if extras.get("scores"):
            cols[1].write("Cross-Encoder 점수")
            cols[1].write(extras["scores"])


# ---------------------------------------------------------------------------
# 사이드바
# ---------------------------------------------------------------------------
st.sidebar.header("⚙️ 공통 설정")
st.sidebar.caption("모든 전략이 같은 벡터스토어 · TOP_N · grounded 체인을 사용합니다. 차이는 Retriever뿐입니다.")

top_n = st.sidebar.slider("TOP_N (최종 문서 수)", 2, 8, 4, 1)
base_k = st.sidebar.slider("BASE_K (Rerank 1차 후보)", 6, 20, 12, 1)
force_reindex = st.sidebar.checkbox("인덱스 강제 재구축", value=False)

with st.spinner("벡터스토어·LLM 준비 중..."):
    try:
        engine = get_engine(top_n, base_k, force_reindex)
        boot_error = None
    except Exception as e:
        engine = None
        boot_error = e

if boot_error is not None:
    st.error(f"엔진 초기화 실패: {boot_error}")
    st.stop()

st.sidebar.success(f"컬렉션 문서 수: {engine.doc_count:,}")
st.sidebar.caption(f"PDF: `{engine.pdf_dir}`")
st.sidebar.caption(f"Chroma: `{engine.persist_dir}`")
st.sidebar.caption(f"임베딩 `{EMBEDDING_MODEL}` · 생성 `{LLM_MODEL}`")
if engine.cohere_key:
    st.sidebar.info("Cohere Rerank 사용 가능")
else:
    st.sidebar.caption("COHERE_API_KEY 없음 → Cross-Encoder Rerank만 사용")

with st.sidebar.expander("초기화 로그"):
    for line in engine.logs:
        st.write(line)

available = engine.strategy_names
default_selected = [s for s in ["Basic", "HyDE", "SelfQuery", "Rerank(CrossEncoder)"] if s in available]
compare_strategies = st.sidebar.multiselect(
    "비교할 전략",
    options=available,
    default=default_selected,
)

# ---------------------------------------------------------------------------
# 본문
# ---------------------------------------------------------------------------
st.markdown('<div class="main-title">🌾 RAG 검색 전략 교체 비교</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">농촌진흥청 연차보고서 RAG · Basic / HyDE / Multi-Query / Self-Query / Rerank · 멀티턴 히스토리 압축</div>',
    unsafe_allow_html=True,
)

tab_single, tab_compare, tab_chat, tab_guide = st.tabs(
    ["① 단일 검색", "② 전략 비교", "③ 멀티턴 대화", "④ 전략 가이드"]
)

# ========================= ① 단일 검색 =========================
with tab_single:
    st.markdown("질문을 하나 넣고 **검색 전략 하나**만 적용해 출처와 grounded 답변을 확인합니다.")
    c1, c2 = st.columns([2, 1])
    with c1:
        sample = st.selectbox("예시 질문", ["(직접 입력)"] + SAMPLE_QUESTIONS, index=1)
        question = st.text_area(
            "질문",
            value="" if sample == "(직접 입력)" else sample,
            height=90,
        )
    with c2:
        strategy = st.radio("검색 전략", available, index=0)
        run_single = st.button("검색 + 답변 생성", type="primary", use_container_width=True)

    if run_single:
        q = question.strip()
        if not q:
            st.warning("질문을 입력하세요.")
        else:
            with st.spinner(f"{strategy} 실행 중..."):
                result = engine.run_strategy(strategy, q, with_extras=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("검색 시간", f"{result['retrieval_time_s']}s")
            m2.metric("생성 시간", f"{result['generation_time_s']}s")
            m3.metric("총 시간", f"{result['total_time_s']}s")
            m4.metric("적중 연도", ", ".join(str(y) for y in result["years_hit"]) or "-")

            render_extras(strategy, result.get("extras") or {})
            st.subheader("검색 문서")
            render_doc_table(result["docs"])
            st.subheader("Grounded 답변")
            st.markdown(result["answer"])

# ========================= ② 전략 비교 =========================
with tab_compare:
    st.markdown("동일 질문·동일 생성 프롬프트에서 **검색 전략만** 바꿔 출처·연도·소요시간을 비교합니다.")
    cmp_sample = st.selectbox("비교용 예시 질문", ["(직접 입력)"] + SAMPLE_QUESTIONS, index=1, key="cmp_sample")
    cmp_question = st.text_area(
        "비교 질문",
        value="" if cmp_sample == "(직접 입력)" else cmp_sample,
        height=80,
        key="cmp_q",
    )
    run_cmp = st.button("선택한 전략 일괄 비교", type="primary")

    if run_cmp:
        q = cmp_question.strip()
        if not q:
            st.warning("질문을 입력하세요.")
        elif not compare_strategies:
            st.warning("사이드바에서 비교할 전략을 하나 이상 선택하세요.")
        else:
            rows = []
            progress = st.progress(0.0, text="비교 시작...")
            for i, name in enumerate(compare_strategies, 1):
                progress.progress(i / len(compare_strategies), text=f"{name} 실행 중...")
                try:
                    rows.append(engine.run_strategy(name, q))
                except Exception as e:
                    rows.append(
                        {
                            "strategy": name,
                            "question": q,
                            "num_docs": 0,
                            "years_hit": [],
                            "sources": [],
                            "retrieval_time_s": None,
                            "generation_time_s": None,
                            "total_time_s": None,
                            "answer": f"오류: {e}",
                            "docs": [],
                            "extras": {"error": str(e)},
                        }
                    )
            progress.empty()
            st.session_state.compare_rows = rows
            st.session_state.compare_question = q

    rows = st.session_state.get("compare_rows")
    if rows:
        st.caption(f"질문: {st.session_state.get('compare_question', '')}")
        summary = pd.DataFrame(
            [
                {
                    "strategy": r["strategy"],
                    "num_docs": r["num_docs"],
                    "years_hit": ", ".join(str(y) for y in r["years_hit"]),
                    "retrieval_s": r["retrieval_time_s"],
                    "generation_s": r["generation_time_s"],
                    "total_s": r["total_time_s"],
                }
                for r in rows
            ]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)

        chart_df = pd.DataFrame(
            {
                "strategy": [r["strategy"] for r in rows if r["total_time_s"] is not None],
                "검색": [r["retrieval_time_s"] for r in rows if r["total_time_s"] is not None],
                "생성": [r["generation_time_s"] for r in rows if r["total_time_s"] is not None],
            }
        )
        if not chart_df.empty:
            st.bar_chart(chart_df.set_index("strategy"))

        for r in rows:
            with st.expander(f"{r['strategy']}  · 연도 {r['years_hit']}  · {r['total_time_s']}s", expanded=False):
                render_extras(r["strategy"], r.get("extras") or {})
                st.markdown("**출처**")
                for src in r["sources"]:
                    st.markdown(f"- {src}")
                render_doc_table(r["docs"])
                st.markdown("**답변**")
                st.markdown(r["answer"])

# ========================= ③ 멀티턴 대화 =========================
with tab_chat:
    st.markdown("대화가 길어지면 오래된 턴을 요약하고 최근 턴 원문만 유지합니다. (노트북 7절과 동일)")

    c1, c2, c3 = st.columns(3)
    chat_strategy = c1.selectbox("대화용 검색 전략", available, index=0, key="chat_strategy")
    keep_recent = c2.slider("최근 원문 유지 턴 수", 1, 6, 2)
    threshold = c3.slider("요약 시작 임계 턴", 2, 10, 3)

    strategy_key = (chat_strategy, top_n, base_k)
    if st.session_state.get("chat_strategy_key") != strategy_key or "chat_mgr" not in st.session_state:
        _init_chat_manager(engine, chat_strategy, keep_recent, threshold)
        st.session_state.chat_strategy_key = strategy_key
    else:
        st.session_state.chat_mgr.keep_recent_turns = keep_recent
        st.session_state.chat_mgr.summarize_threshold = threshold

    b1, b2, c_reset = st.columns([3, 1.2, 1])
    with b1:
        preset = st.selectbox("멀티턴 예시 질문", ["(선택)"] + MULTI_TURN_QUESTIONS)
    with b2:
        send_preset = st.button("예시 질문 보내기", use_container_width=True)
    with c_reset:
        if st.button("대화 초기화", use_container_width=True):
            _init_chat_manager(engine, chat_strategy, keep_recent, threshold)
            st.rerun()

    mgr = st.session_state.chat_mgr
    m1, m2, m3, m4 = st.columns(4)
    last = st.session_state.chat_metrics[-1] if st.session_state.chat_metrics else None
    m1.metric("대화 턴", len(mgr.turns))
    m2.metric("히스토리 토큰", last["history_tokens"] if last else 0)
    m3.metric("프롬프트 토큰", last["prompt_tokens"] if last else 0)
    m4.metric("압축 횟수", len(mgr.compression_events))

    if mgr.summary:
        with st.expander("현재 누적 요약", expanded=False):
            st.write(mgr.summary)

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("meta"):
                meta = msg["meta"]
                st.caption(
                    f"prompt {meta['prompt_tokens']} tok · history {meta['history_tokens']} tok · docs {meta['doc_tokens']} tok"
                )

    user_q = st.chat_input("연차보고서에 대해 이어서 질문하세요")
    if send_preset and preset != "(선택)":
        user_q = preset

    if user_q:
        st.session_state.chat_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("검색 후 답변 생성 중..."):
                result = mgr.ask(user_q)
            st.markdown(result["answer"])
            st.caption(
                f"prompt {result['prompt_tokens']} tok · history {result['history_tokens']} tok · docs {result['doc_tokens']} tok"
            )
            with st.expander("이번 턴 검색 문서"):
                render_doc_table(result["docs"])
        st.session_state.chat_messages.append(
            {"role": "assistant", "content": result["answer"], "meta": result}
        )
        st.session_state.chat_metrics.append(result)
        if result.get("just_compressed"):
            st.toast("오래된 대화를 요약해 컨텍스트를 압축했습니다.")
        st.rerun()

# ========================= ④ 가이드 =========================
with tab_guide:
    st.markdown("동일 코퍼스·동일 생성 체인에서 검색 전략만 바꿨을 때의 선택 가이드입니다.")
    st.dataframe(pd.DataFrame(STRATEGY_GUIDE), use_container_width=True, hide_index=True)
    st.markdown(
        """
**실무 파이프라인 예**

`Self-Query(연도 필터) → 넓은 k → Rerank(TOP_N) → grounded 생성` + 멀티턴 요약 압축

**공정 비교를 지킨 점**
- 임베딩(`text-embedding-3-small`) · 생성(`gpt-4o-mini`) · TOP_N · 답변 프롬프트 고정
- 차이는 Retriever/검색 전략과 (Rerank의) 1차 `BASE_K`뿐
- 답변은 검색 문맥에만 근거 (grounded). HyDE 가상 문서는 검색에만 사용
"""
    )
