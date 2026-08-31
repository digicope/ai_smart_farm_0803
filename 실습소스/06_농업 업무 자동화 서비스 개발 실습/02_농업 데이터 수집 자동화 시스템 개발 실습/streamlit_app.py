# -*- coding: utf-8 -*-
"""농업 데이터 수집 자동화 Agent Streamlit 앱.

실행 (이 폴더에서):
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from farm_collect_core import (
    ALLOWED_CROPS,
    DATA_DIR,
    DB_PATH,
    MISSING_CROPS,
    SAMPLE_QUESTIONS,
    SOURCE_FILES,
    TOOL_LABELS,
    ask_agent,
    build_runtime,
    connect_db,
    init_schema,
    load_config,
    table_count,
)

st.set_page_config(
    page_title="농업 데이터 수집 Agent",
    page_icon="🚜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .stApp { background: linear-gradient(180deg, #f3f7f1 0%, #eef4ea 40%, #f7f5ef 100%); }
      .hero {
        background: linear-gradient(135deg, #1f5c3a 0%, #3d8b5a 55%, #6b8f3a 100%);
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
      .chip-muted {
        display: inline-block;
        background: #f3eee6;
        color: #6b5340;
        border: 1px solid #e0d4c4;
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


@st.cache_resource(show_spinner="수집 Agent를 준비하는 중...")
def get_runtime():
    return build_runtime()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""


def warehouse_counts() -> dict[str, int]:
    conn = connect_db()
    try:
        init_schema(conn)
        return {
            name: table_count(conn, name)
            for name in ("weather", "farm", "growth", "pest_sites", "pest_info")
        }
    finally:
        conn.close()


def csv_cache_flags() -> dict[str, bool]:
    return {source: path.exists() for source, path in SOURCE_FILES.items()}


def render_sidebar(runtime: dict) -> None:
    cfg = load_config()
    counts = warehouse_counts()
    cached = csv_cache_flags()
    with st.sidebar:
        st.markdown("### 수집 설정")
        region = cfg["region_focus"]
        st.markdown(f"**초점 지역:** {region['sido']} {region['sigungu']}")
        st.caption(region["note"])
        st.markdown("**있는 작목**")
        st.markdown(
            " ".join(f'<span class="chip">{crop}</span>' for crop in ALLOWED_CROPS),
            unsafe_allow_html=True,
        )
        st.markdown("**없는 작목 (대체 금지)**")
        st.markdown(
            " ".join(f'<span class="chip-muted">{crop}</span>' for crop in MISSING_CROPS),
            unsafe_allow_html=True,
        )
        st.divider()
        st.markdown("### 창고 현황")
        st.caption(str(DB_PATH))
        c1, c2 = st.columns(2)
        c1.metric("기상", f"{counts['weather']:,}")
        c2.metric("농가", f"{counts['farm']:,}")
        c3, c4 = st.columns(2)
        c3.metric("생육", f"{counts['growth']:,}")
        c4.metric("예찰·병해충", f"{counts['pest_sites'] + counts['pest_info']:,}")
        st.caption(
            "CSV 캐시: "
            + ", ".join(
                f"{src}{'○' if ok else '×'}" for src, ok in cached.items()
            )
        )
        st.divider()
        st.markdown("### 경보 임계값")
        th = cfg["alert_thresholds"]
        st.write(
            f"습도 {th['humidity_high_pct']}% · "
            f"강수 {th['rainfall_heavy_mm']}mm · "
            f"이상치 {th['rainfall_sensor_outlier_mm']}mm · "
            f"고온 {th['tmax_hot_c']}℃"
        )
        st.divider()
        st.markdown("### 시스템")
        st.success("OPENAI_API_KEY 로드 완료 (값은 표시하지 않습니다)")
        st.caption(f"캐시 폴더: {DATA_DIR.name}/  ·  Open API serviceKey 없음")
        if st.button("대화 지우기", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = ""
            st.rerun()
        st.caption(
            "시나리오 B는 포털에서 CSV/ZIP을 받습니다. "
            "기상 원본은 약 85MB이며, 정리본이 있으면 생략합니다."
        )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>🚜 농업 데이터 수집 자동화 Agent</h1>
          <p>공공데이터포털 파일데이터(CSV/ZIP)를 Tool이 직접 내려받고,
          SQLite 창고에 적재·조회·경보합니다. Open API 인증키는 쓰지 않습니다.</p>
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


def render_sources() -> None:
    cfg = load_config()
    with st.expander("수집 대상 (COLLECTION_CONFIG)", expanded=False):
        df = pd.DataFrame(cfg["sources"])[
            ["source_id", "name", "file", "year", "provider", "portal"]
        ]
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("고추 생육은 전국 자료입니다. 기상만 완주로 필터하세요.")


def render_tool_pills(tools: list[str]) -> None:
    labels = [TOOL_LABELS.get(name, name) for name in tools]
    html = (
        " ".join(f'<span class="tool-pill">호출: {label}</span>' for label in labels)
        if labels
        else '<span class="tool-pill">도구 호출 없음</span>'
    )
    st.markdown(html, unsafe_allow_html=True)


def render_trace(trace: list[dict], expanded: bool = False) -> None:
    if not trace:
        return
    with st.expander("도구 호출 과정 보기", expanded=expanded):
        for step in trace:
            if step.get("label") == "결과":
                st.caption(f"← {step.get('preview', '')}")
            elif "args" in step:
                st.write(f"→ **{step['label']}** `{step.get('args', {})}`")


def render_history() -> None:
    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            if item["role"] == "assistant":
                render_tool_pills(item.get("tools") or [])
            st.markdown(item["content"])
            if item["role"] == "assistant":
                render_trace(item.get("trace") or [])


def handle_question(runtime: dict, question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("포털 자료와 창고를 확인하는 중... (전체 수집은 시간이 걸릴 수 있습니다)"):
            result = ask_agent(runtime, question)
        render_tool_pills(result["tools"])
        st.markdown(result["answer"])
        render_trace(result["trace"], expanded=True)
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result["answer"],
            "tools": result["tools"],
            "trace": result["trace"],
        }
    )
    st.rerun()


def main() -> None:
    init_state()
    try:
        runtime = get_runtime()
    except Exception as exc:
        st.error(f"Agent를 준비하지 못했습니다: {exc}")
        st.info(r"`C:\env\.env`의 OPENAI_API_KEY 를 확인하세요. Open API serviceKey는 필요 없습니다.")
        return

    render_sidebar(runtime)
    render_hero()
    render_samples()
    render_sources()
    st.divider()
    render_history()

    typed = st.chat_input("수집 대상, 포털 다운로드, 완주 기상, 고추 생육, 경보를 물어보세요")
    question = st.session_state.pending_question or typed
    if st.session_state.pending_question:
        st.session_state.pending_question = ""
    if question:
        handle_question(runtime, question)


if __name__ == "__main__":
    main()
