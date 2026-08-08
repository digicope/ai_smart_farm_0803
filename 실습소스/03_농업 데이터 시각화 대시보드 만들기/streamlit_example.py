"""
농업 데이터 시각화 대시보드 (Streamlit 예제)

실행 방법
--------
1) 터미널에서 이 파일이 있는 폴더로 이동
2) 아래 명령 실행:
       streamlit run sreamlit_example.py
3) 브라우저가 자동으로 열리면 대시보드를 확인

중지 방법
--------
- 터미널에서 Ctrl + C
"""

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ------------------------------------------------------------
# 사용법: set_page_config()
# - 페이지 제목, 아이콘, 레이아웃(wide/centered)을 설정합니다.
# - 반드시 다른 st.* 호출보다 먼저, 그리고 한 번만 호출하세요.
# ------------------------------------------------------------
st.set_page_config(
    page_title="농업 데이터 대시보드",
    page_icon="🌾",
    layout="wide",
)


@st.cache_data
def load_agriculture_data() -> pd.DataFrame:
    """가상 농업 데이터를 생성합니다.

    사용법:
        - @st.cache_data : 같은 입력이면 결과를 캐시해 재계산을 줄입니다.
        - 실제 업무에서는 아래 생성 로직을 pd.read_csv / DB 조회로 바꾸면 됩니다.
    """
    rng = np.random.default_rng(42)
    months = [f"{m}월" for m in range(1, 13)]
    crops = ["벼", "옥수수", "감자", "토마토"]

    rows = []
    for crop in crops:
        base = {"벼": 40, "옥수수": 55, "감자": 30, "토마토": 25}[crop]
        for i, month in enumerate(months, start=1):
            harvest = base + 20 * np.sin((i - 3) / 12 * 2 * np.pi) + rng.normal(0, 3)
            temp = 12 + 15 * np.sin((i - 3) / 12 * 2 * np.pi) + rng.normal(0, 1)
            rain = 60 + 40 * np.sin((i - 1) / 12 * 2 * np.pi) + rng.normal(0, 8)
            rows.append(
                {
                    "월": month,
                    "월번호": i,
                    "작물": crop,
                    "수확량_톤": round(max(harvest, 5), 1),
                    "기온_C": round(temp, 1),
                    "강수량_mm": round(max(rain, 5), 1),
                }
            )
    return pd.DataFrame(rows)


# 사용법: 데이터 로드 (캐시됨)
df = load_agriculture_data()

# ------------------------------------------------------------
# 사용법: title / caption
# - st.title()  : 페이지 최상단 큰 제목
# - st.caption(): 보조 설명(작은 회색 텍스트)
# ------------------------------------------------------------
st.title("농업 데이터 시각화 대시보드")
st.caption("Streamlit + Pandas + Plotly 예제 | 작물·월 필터로 수확량과 기상을 탐색합니다.")

# ------------------------------------------------------------
# 사용법: sidebar 위젯
# - st.sidebar.selectbox(label, options) : 드롭다운 선택
# - st.sidebar.multiselect(...)          : 다중 선택
# - st.sidebar.slider(min, max, value)   : 범위/숫자 슬라이더
# 반환값 = 사용자가 현재 선택한 값 (위젯이 바뀌면 스크립트가 다시 실행됨)
# ------------------------------------------------------------
st.sidebar.header("필터")

selected_crops = st.sidebar.multiselect(
    "작물 선택",
    options=sorted(df["작물"].unique()),
    default=sorted(df["작물"].unique()),  # 사용법: default로 초기 선택값 지정
    help="비교할 작물을 하나 이상 선택하세요.",  # 사용법: help는 ? 아이콘 툴팁
)

month_start, month_end = st.sidebar.slider(
    "월 범위",
    min_value=1,
    max_value=12,
    value=(1, 12),  # 사용법: 튜플이면 범위 슬라이더가 됩니다
)

chart_type = st.sidebar.radio(
    "차트 종류",
    options=["선 그래프", "막대 그래프", "산점도"],
    index=0,  # 사용법: 기본으로 첫 번째 옵션 선택
)

show_raw = st.sidebar.checkbox("원본 데이터 보기", value=False)

# 사용법: 선택이 비어 있으면 안내 메시지를 보여주고 중단
if not selected_crops:
    st.warning("사이드바에서 작물을 하나 이상 선택해 주세요.")
    st.stop()  # 사용법: st.stop() 이후 코드는 실행되지 않습니다

# 사용법: 사이드바 선택값으로 데이터 필터링
filtered = df[
    (df["작물"].isin(selected_crops))
    & (df["월번호"].between(month_start, month_end))
].copy()

# ------------------------------------------------------------
# 사용법: metric + columns
# - st.columns(n) : 화면을 n개 열로 나눕니다
# - st.metric(label, value, delta=...) : KPI 카드
# ------------------------------------------------------------
st.subheader("핵심 지표 (KPI)")
k1, k2, k3, k4 = st.columns(4)

avg_harvest = filtered["수확량_톤"].mean()
total_harvest = filtered["수확량_톤"].sum()
avg_temp = filtered["기온_C"].mean()
avg_rain = filtered["강수량_mm"].mean()

k1.metric("평균 수확량", f"{avg_harvest:.1f} 톤")
k2.metric("총 수확량", f"{total_harvest:.1f} 톤")
k3.metric("평균 기온", f"{avg_temp:.1f} °C")
k4.metric("평균 강수량", f"{avg_rain:.1f} mm")

st.divider()  # 사용법: 시각적 구분선

# ------------------------------------------------------------
# 사용법: Plotly 차트 + st.plotly_chart
# - use_container_width=True 이면 컨테이너 너비에 맞춰 늘어납니다
# ------------------------------------------------------------
left, right = st.columns((2, 1))

with left:
    st.subheader("월별 수확량")

    if chart_type == "선 그래프":
        # 사용법: px.line — 시간 추이(선) 시각화
        fig = px.line(
            filtered,
            x="월",
            y="수확량_톤",
            color="작물",
            markers=True,
            title="작물별 월간 수확량 추이",
        )
    elif chart_type == "막대 그래프":
        # 사용법: px.bar — 범주 비교(막대) 시각화
        fig = px.bar(
            filtered,
            x="월",
            y="수확량_톤",
            color="작물",
            barmode="group",
            title="작물별 월간 수확량 비교",
        )
    else:
        # 사용법: px.scatter — 두 변수 관계(산점도)
        fig = px.scatter(
            filtered,
            x="기온_C",
            y="수확량_톤",
            color="작물",
            size="강수량_mm",
            hover_data=["월"],
            title="기온 vs 수확량 (점 크기 = 강수량)",
        )

    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("작물별 합계")
    by_crop = (
        filtered.groupby("작물", as_index=False)["수확량_톤"]
        .sum()
        .sort_values("수확량_톤", ascending=False)
    )
    fig_pie = px.pie(
        by_crop,
        names="작물",
        values="수확량_톤",
        title="선택 기간 수확량 비중",
        hole=0.35,  # 사용법: hole > 0 이면 도넛 차트
    )
    st.plotly_chart(fig_pie, use_container_width=True)

# ------------------------------------------------------------
# 사용법: dataframe / download_button
# - st.dataframe : 스크롤·정렬 가능한 표
# - st.download_button : CSV 등 파일 다운로드 버튼
# ------------------------------------------------------------
st.subheader("요약 테이블")
summary = (
    filtered.groupby("작물", as_index=False)
    .agg(
        평균수확량=("수확량_톤", "mean"),
        총수확량=("수확량_톤", "sum"),
        평균기온=("기온_C", "mean"),
        평균강수량=("강수량_mm", "mean"),
    )
    .round(1)
)
st.dataframe(summary, use_container_width=True)

csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")  # 사용법: 엑셀 한글용 utf-8-sig
st.download_button(
    label="필터된 데이터 CSV 다운로드",
    data=csv_bytes,
    file_name="agriculture_filtered.csv",
    mime="text/csv",
)

if show_raw:
    st.subheader("원본(필터) 데이터")
    st.dataframe(filtered, use_container_width=True)

# 사용법: expander — 접었다 펼 수 있는 도움말 영역
with st.expander("Streamlit 사용 팁"):
    st.markdown(
        """
        - 위젯 값이 바뀌면 **스크립트 전체가 다시 실행**됩니다.
        - 무거운 데이터 로딩은 `@st.cache_data`로 감싸세요.
        - 레이아웃은 `st.sidebar`, `st.columns`, `st.tabs`로 구성합니다.
        - 배포는 Streamlit Community Cloud 또는 Docker/서버에 `streamlit run`으로 가능합니다.
        """
    )
