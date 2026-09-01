# -*- coding: utf-8 -*-
"""AI 농업 정보 검색 Agent Streamlit 앱.

실행:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from agri_search_core import SAMPLE_QUESTIONS, TOOL_LABELS, ask_agent, build_runtime

st.set_page_config(
    page_title="농업 정보 검색 Agent",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f3f7f1 0%, #eef4ea 40%, #f7f5ef 100%); }
      .hero {
        background: linear-gradient(135deg, #2f6b3a 0%, #4a8c4a 55%, #6b8f3a 100%);
        color: #fff;
        padding: 1.3rem 1.5rem;
        border-radius: 18px;
        margin-bottom: 1rem;
      }
      .hero h1 { font-size: 1.7rem; margin: 0 0 0.35rem 0; }
      .hero p { margin: 0; opacity: 0.92; }
      .chip {
        display: inline-block;
        background: #e7f2e4;
        color: #245c2e;
        border: 1px solid #c5ddc0;
        border-radius: 999px;
        padding: 0.15rem 0.7rem;
        margin: 0.15rem 0.25rem 0.15rem 0;
        font-size: 0.85rem;
      }
      .tool-pill {
        display: inline-block;
        background: #fff7e6;
        color: #8a5a00;
        border: 1px solid #f0d9a6;
        border-radius: 999px;
        padding: 0.12rem 0.65rem;
        margin-right: 0.3rem;
        font-size: 0.82rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="PDF 인덱스와 Agent를 준비하는 중...")
def get_runtime():
    return build_runtime()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""


def render_sidebar(runtime: dict) -> None:
    with st.sidebar:
        st.markdown("### 검색 범위")
        st.markdown("**PDF**")
        st.markdown(
            " ".join(f'<span class="chip">{c}</span>' for c in runtime["crops_pdf"]),
            unsafe_allow_html=True,
        )
        st.markdown("**노지 CSV**")
        st.markdown(
            " ".join(f'<span class="chip">{c}</span>' for c in runtime["crops_field"]),
            unsafe_allow_html=True,
        )
        st.markdown("**스마트팜 CSV**")
        st.markdown(
            " ".join(f'<span class="chip">{c}</span>' for c in runtime["crops_smart"]),
            unsafe_allow_html=True,
        )
        st.divider()
        st.success("OPENAI_API_KEY 로드 완료 (값은 표시하지 않습니다)")
        st.metric("작물 길잡이 벡터", f"{runtime['vector_count']:,}")
        st.caption(f"PDF {runtime['pdf_count']}종 · FAISS persist 재사용")
        st.divider()
        if st.button("대화 지우기", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = ""
            st.rerun()
        st.caption("재배법은 PDF, 숫자는 CSV, 최신 이슈는 웹 검색만 근거로 답합니다.")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🌾 농업 정보 검색 Agent</h1>
          <p>농업기술길잡이 RAG, 노지·스마트팜 실측 CSV, 웹 검색을 질문 의도에 맞게 골라 답합니다.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_samples() -> None:
    st.markdown("**실습 시나리오로 물어보기**")
    cols = st.columns(len(SAMPLE_QUESTIONS))
    for col, sample in zip(cols, SAMPLE_QUESTIONS):
        if col.button(sample["label"], use_container_width=True):
            st.session_state.pending_question = sample["text"]
            st.rerun()


def render_history() -> None:
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            if item["role"] == "assistant" and item.get("tools"):
                labels = [TOOL_LABELS.get(name, name) for name in item["tools"]]
                st.markdown(
                    " ".join(f'<span class="tool-pill">호출: {label}</span>' for label in labels)
                    or '<span class="tool-pill">도구 호출 없음</span>',
                    unsafe_allow_html=True,
                )
            st.markdown(item["content"])
            if item["role"] == "assistant" and item.get("trace"):
                with st.expander("도구 호출 과정 보기"):
                    for step in item["trace"]:
                        if step.get("label") == "결과":
                            st.caption(f"← {step.get('preview', '')}")
                        elif "args" in step:
                            st.write(f"→ **{step['label']}** `{step.get('args', {})}`")


def handle_question(runtime: dict, question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("PDF·CSV·웹 검색을 확인하는 중..."):
            result = ask_agent(runtime, question)
        if result["tools"]:
            labels = [TOOL_LABELS.get(name, name) for name in result["tools"]]
            st.markdown(
                " ".join(f'<span class="tool-pill">호출: {label}</span>' for label in labels),
                unsafe_allow_html=True,
            )
        st.markdown(result["answer"])
        if result["trace"]:
            with st.expander("도구 호출 과정 보기", expanded=True):
                for step in result["trace"]:
                    if step.get("label") == "결과":
                        st.caption(f"← {step.get('preview', '')}")
                    elif "args" in step:
                        st.write(f"→ **{step['label']}** `{step.get('args', {})}`")
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "tools": result["tools"],
            "trace": result["trace"],
        }
    )


def main() -> None:
    init_state()
    try:
        runtime = get_runtime()
    except Exception as exc:
        st.error(f"Agent를 준비하지 못했습니다: {exc}")
        st.info(r"`C:\env\.env`의 OPENAI_API_KEY, pdf/ 길잡이, 노지·스마트팜 ZIP을 확인하세요.")
        return

    render_sidebar(runtime)
    render_hero()
    render_samples()
    st.divider()
    render_history()

    typed = st.chat_input("고구마 정식, 노지 고추 생육, 스마트팜 토마토, 최근 딸기 이슈 등을 물어보세요")
    question = st.session_state.pending_question or typed
    if st.session_state.pending_question:
        st.session_state.pending_question = ""
    if question:
        handle_question(runtime, question)


main()
