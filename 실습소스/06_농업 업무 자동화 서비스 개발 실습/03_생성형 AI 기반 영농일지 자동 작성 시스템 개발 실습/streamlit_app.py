# -*- coding: utf-8 -*-
"""생성형 AI 영농일지 자동 작성 Agent Streamlit 앱.

실행 (이 폴더에서):
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from farm_diary_core import (
    ALLOWED_CROPS,
    DB_PATH,
    MISSING_CROPS,
    SAMPLE_MEMOS,
    SAMPLE_QUESTIONS,
    TOOL_LABELS,
    ask_agent,
    build_runtime,
    load_diary_table,
    reset_diary_table,
    sample_weather_rows,
    warehouse_overview,
)

st.set_page_config(
    page_title="영농일지 자동 작성 Agent",
    page_icon="📒",
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


@st.cache_resource(show_spinner="영농일지 Agent와 창고를 준비하는 중...")
def get_runtime():
    return build_runtime()


def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""


def weather_brief(raw: str) -> str:
    try:
        block = json.loads(raw or "{}") or {}
    except json.JSONDecodeError:
        return ""
    reason = block.get("missing_reason") or ""
    if reason:
        return reason[:48]
    parts = []
    if block.get("tmin_c") is not None:
        parts.append(f"{block.get('tmin_c')}~{block.get('tmax_c')}℃")
    if block.get("rainfall_mm") is not None:
        parts.append(f"{block.get('rainfall_mm')}mm")
    if block.get("station"):
        parts.append(str(block["station"]))
    return " / ".join(parts)


def diary_table_view() -> pd.DataFrame:
    df = load_diary_table()
    if df.empty:
        return df
    view = df.copy()
    view["기상"] = view["weather_json"].map(weather_brief)
    return view[["diary_id", "work_date", "crop", "region", "기상", "saved_at"]]


def render_sidebar() -> None:
    overview = warehouse_overview()
    diaries = load_diary_table()
    counts = {row["table"]: int(row["건수"]) for _, row in overview.iterrows()}
    with st.sidebar:
        st.markdown("### 일지 설정")
        st.markdown("**초점 지역:** 전북 완주군")
        st.caption("기상은 2023년 반교리 우선. 농가·생육은 2024년 참고입니다.")
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
        c1.metric("기상", f"{counts.get('weather', 0):,}")
        c2.metric("농가", f"{counts.get('farm', 0):,}")
        c3, c4 = st.columns(2)
        c3.metric("생육", f"{counts.get('growth', 0):,}")
        c4.metric("병해충", f"{counts.get('pest_info', 0):,}")
        st.metric("저장된 영농일지", f"{len(diaries):,}")
        st.divider()
        st.markdown("### 시스템")
        st.success("OPENAI_API_KEY 로드 완료 (값은 표시하지 않습니다)")
        st.caption("Open API serviceKey 없음 · 기상청 실API 대신 창고 2023 기상 사용")
        if st.button("대화 지우기", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pending_question = ""
            st.rerun()
        if st.button("저장된 일지 비우기", use_container_width=True):
            reset_diary_table()
            st.session_state.messages = []
            st.session_state.pending_question = ""
            st.rerun()
        st.caption("일지 숫자는 창고와 메모만 근거로 채웁니다. 약제명·수확량은 지어내지 않습니다.")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
          <h1>📒 생성형 AI 영농일지 자동 작성</h1>
          <p>구어체 메모를 해석하고, 완주 2023 기상·노지 농가·생육·병해충 창고를 근거로
          구조화 일지를 만든 뒤 SQLite에 저장합니다.</p>
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


def render_memo_box() -> None:
    with st.expander("직접 메모 넣기", expanded=False):
        st.caption("작업일은 2023-08-10 / 08-15 / 08-18 처럼 창고에 있는 날짜가 대조하기 쉽습니다.")
        examples = {item["id"] + ". " + item["title"]: item["memo"] for item in SAMPLE_MEMOS}
        picked = st.selectbox(
            "샘플 메모 불러오기",
            ["(직접 입력)"] + list(examples),
            key="memo_pick",
        )
        if picked != "(직접 입력)" and st.session_state.get("memo_pick_applied") != picked:
            st.session_state.memo_input = examples[picked]
            st.session_state.memo_pick_applied = picked
        memo = st.text_area("구어체 영농 메모", height=110, key="memo_input")
        if st.button("이 메모로 일지 작성·저장", use_container_width=True):
            text = (memo or "").strip()
            if text:
                st.session_state.pending_question = (
                    f"다음 메모를 영농일지로 작성하고 저장해줘.\n{text}"
                )
                st.rerun()


def render_reference() -> None:
    left, right = st.columns(2)
    with left:
        with st.expander("창고 테이블 건수·기간", expanded=False):
            st.dataframe(warehouse_overview(), use_container_width=True, hide_index=True)
            st.caption("고추 생육은 전국 자료입니다. 토마토·딸기·수박·참외는 없습니다.")
    with right:
        with st.expander("시나리오용 기상 샘플 (반교리·이서면)", expanded=False):
            weather = sample_weather_rows()
            st.dataframe(weather, use_container_width=True, hide_index=True)
            st.caption("이서면 강수가 400mm를 넘으면 폭우로 단정하지 않습니다. 일지는 반교리 값을 씁니다.")


def render_saved_diaries() -> None:
    st.markdown("**저장된 영농일지**")
    view = diary_table_view()
    if view.empty:
        st.info("아직 저장된 일지가 없습니다. 시나리오 A~C 또는 메모 작성으로 저장해 보세요.")
        return
    st.dataframe(view, use_container_width=True, hide_index=True)
    full = load_diary_table()
    options = [
        f"{row.diary_id} | {row.work_date} | {row.crop}"
        for row in full.itertuples()
    ]
    picked = st.selectbox("일지 본문 보기", options, index=len(options) - 1)
    diary_id = int(str(picked).split("|", 1)[0].strip())
    row = full.loc[full["diary_id"] == diary_id].iloc[0]
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**문장형 일지**")
        st.write(row["narrative_ko"] or "(없음)")
        st.markdown("**작업**")
        st.code(row["work_items_json"] or "[]", language="json")
    with c2:
        st.markdown("**기상 근거**")
        st.code(row["weather_json"] or "{}", language="json")
        st.markdown("**생육 / 병해충 / 다음 계획**")
        st.write(f"생육: {row['growth_note'] or '-'}")
        st.write(f"병해충: {row['pest_note'] or '-'}")
        st.write(f"특이사항: {row['special_note'] or '-'}")
        st.write(f"다음 계획: {row['next_plan'] or '-'}")


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
                args = step.get("args") or {}
                short = {
                    key: (str(val)[:120] + "…" if len(str(val)) > 120 else val)
                    for key, val in args.items()
                }
                st.write(f"→ **{step['label']}** `{short}`")


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
        with st.spinner("메모를 해석하고 창고 근거로 일지를 작성하는 중..."):
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
        st.info(
            r"`C:\env\.env`의 OPENAI_API_KEY 와 "
            r"`C:\env\farm_warehouse\farm_warehouse.db` 를 확인하세요. "
            "창고가 비면 02 폴더 CSV를 자동 적재합니다."
        )
        return

    render_sidebar()
    render_hero()
    render_samples()
    render_memo_box()
    render_reference()
    st.divider()
    render_saved_diaries()
    st.divider()
    render_history()

    typed = st.chat_input(
        "예: 2023년 8월 15일 완주 고추밭 일지를 작성하고 저장해줘"
    )
    question = st.session_state.pending_question or typed
    if st.session_state.pending_question:
        st.session_state.pending_question = ""
    if question:
        handle_question(runtime, question)


if __name__ == "__main__":
    main()
