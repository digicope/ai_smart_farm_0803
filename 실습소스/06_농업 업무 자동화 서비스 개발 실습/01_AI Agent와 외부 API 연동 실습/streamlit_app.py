# -*- coding: utf-8 -*-
"""농업 의사결정 Agent Streamlit 앱.

실행:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

from agri_agent_core import (
    ALLOWED_CROPS,
    LOCATION_NAME,
    SAMPLE_QUESTIONS,
    TOOL_LABELS,
    ask_agent,
    build_runtime,
)

st.set_page_config(
    page_title="농업 의사결정 Agent",
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
      div[data-testid="stChatMessage"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Agent와 작물 길잡이 인덱스를 준비하는 중...")
def get_runtime():
    return build_runtime()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""


def render_sidebar(runtime: dict) -> None:
    with st.sidebar:
        st.markdown("### 조회 설정")
        st.markdown(f"**지역:** {runtime['location']}")
        st.caption("단기예보는 격자, 중기예보는 구역코드를 사용합니다. 좌표·코드는 답변에 출력하지 않습니다.")
        st.markdown("**검색 가능한 작물**")
        st.markdown(
            " ".join(f'<span class="chip">{crop}</span>' for crop in runtime["crops"]),
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### 시스템")
        st.success("API 키 로드 완료 (값은 표시하지 않습니다)")
        st.metric("작물 길잡이 벡터", f"{runtime['vector_count']:,}")
        st.caption(f"PDF {runtime['pdf_count']}종 · FAISS persist 재사용")
        st.divider()
        if st.button("대화 지우기", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = ""
            st.rerun()
        st.caption("날씨는 기상청 실API, 재배법은 농업기술길잡이 PDF만 근거로 답합니다.")


def render_hero() -> None:
    st.markdown(
        f"""
        <div class="hero">
          <h1>🌾 농업 의사결정 Agent</h1>
          <p>기상청 단기·중기예보와 농촌진흥청 농업기술길잡이를 함께 보고, {LOCATION_NAME} 기준으로 농작업 조언을 만듭니다.</p>
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
        with st.spinner("예보와 길잡이를 확인하는 중..."):
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
        st.info("`C:\\env\\.env` 키와 같은 폴더의 농업기술길잡이 PDF, `C:\\env\\crop_guide_faiss` 인덱스를 확인하세요.")
        return

    render_sidebar(runtime)
    render_hero()
    render_samples()
    st.divider()
    render_history()

    typed = st.chat_input(
        f"{LOCATION_NAME} 날씨나 {', '.join(ALLOWED_CROPS)} 재배를 물어보세요"
    )
    question = st.session_state.pending_question or typed
    if st.session_state.pending_question:
        st.session_state.pending_question = ""
    if question:
        handle_question(runtime, question)


main()
