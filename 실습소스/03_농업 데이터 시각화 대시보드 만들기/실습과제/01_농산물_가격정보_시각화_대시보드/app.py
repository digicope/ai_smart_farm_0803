"""
농산물 가격 정보 시각화 대시보드 (실습 과제 답안)

실행 방법
--------
1) 이 파일이 있는 폴더로 이동
2) 필요 패키지 설치:
       pip install pandas openpyxl plotly streamlit
3) 앱 실행:
       streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------
# 페이지 설정 (다른 st.* 호출보다 먼저, 한 번만)
# ------------------------------------------------------------
st.set_page_config(
    page_title="농산물 가격 정보 대시보드",
    page_icon="🥬",
    layout="wide",
)

DATA_DIR = Path(__file__).resolve().parent

# 농산물 이름 → Excel 파일명
CROP_FILES = {
    "딸기": "딸기_가격정보.xlsx",
    "배추": "배추_가격정보.xlsx",
    "사과": "사과_가격정보.xlsx",
    "수박": "수박_가격정보.xlsx",
    "쌀": "쌀_가격정보.xlsx",
}


@st.cache_data
def load_price_data(crop_name: str) -> pd.DataFrame:
    """선택한 농산물의 KAMIS Excel 가격 데이터를 불러와 정리합니다.

    Excel 구조 (KAMIS 기간별 다운로드):
      - 0행: 제목
      - 1행: 컬럼명 (날짜, 가격, 등락률)
      - 2~4행: 전월/전년/평년 요약
      - 5행~: 일자별 가격
    """
    file_path = DATA_DIR / CROP_FILES[crop_name]
    raw = pd.read_excel(file_path, header=1)

    df = raw.copy()
    df.columns = ["날짜", "가격", "등락률"]

    # 일자 형식(YYYY.MM.DD) 행만 사용
    df = df[df["날짜"].astype(str).str.match(r"^\d{4}\.\d{2}\.\d{2}$")].copy()

    df["날짜"] = pd.to_datetime(df["날짜"], format="%Y.%m.%d")
    df["가격"] = (
        df["가격"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .astype(float)
    )
    df["등락률"] = pd.to_numeric(df["등락률"], errors="coerce")

    # 오래된 날짜 → 최신 날짜 순으로 정렬 (라인 차트용)
    df = df.sort_values("날짜").reset_index(drop=True)
    return df


@st.cache_data
def load_all_average_prices() -> pd.DataFrame:
    """5개 농산물의 평균 가격을 비교용으로 계산합니다."""
    rows = []
    for crop in CROP_FILES:
        data = load_price_data(crop)
        rows.append({"농산물": crop, "평균가격": round(data["가격"].mean(), 0)})
    return pd.DataFrame(rows)


# 월별 등락률 그래프에 고정으로 표시할 연월 (3월~7월)
MONTH_RANGE = [f"2026-{m:02d}" for m in range(3, 8)]


@st.cache_data
def load_all_monthly_change_rates() -> pd.DataFrame:
    """5개 농산물의 월별 등락률(전월 대비)을 3월~7월 기준으로 계산합니다.

    - 3월은 기준월로 등락률 0%를 표시합니다.
    - 데이터가 없는 월(예: 딸기 6~7월)은 NaN으로 두어 선이 끊깁니다.
    """
    frames = []
    for crop in CROP_FILES:
        data = load_price_data(crop).copy()
        data["연월"] = data["날짜"].dt.to_period("M").astype(str)
        monthly = (
            data.groupby("연월", as_index=False)["가격"]
            .mean()
            .sort_values("연월")
        )
        monthly["등락률"] = monthly["가격"].pct_change() * 100

        # 3월~7월 전체 구간으로 맞춤
        monthly = monthly.set_index("연월").reindex(MONTH_RANGE).reset_index()
        # 기준월(3월)은 전월 대비 값이 없으므로 0%로 표시
        march_has_price = monthly.loc[monthly["연월"] == "2026-03", "가격"].notna().any()
        if march_has_price:
            monthly.loc[monthly["연월"] == "2026-03", "등락률"] = 0.0

        monthly["농산물"] = crop
        frames.append(monthly[["연월", "농산물", "등락률"]])

    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------
# 화면 구성
# ------------------------------------------------------------
st.title("농산물 가격 정보 대시보드")
st.caption("KAMIS 농산물 가격 데이터 · Streamlit + Pandas + Plotly")

selected_crop = st.selectbox("농산물 선택", options=list(CROP_FILES.keys()))

df = load_price_data(selected_crop)

avg_price = df["가격"].mean()
max_price = df["가격"].max()
min_price = df["가격"].min()
max_date = df.loc[df["가격"].idxmax(), "날짜"]
min_date = df.loc[df["가격"].idxmin(), "날짜"]

# 전월 대비 가격 변화율 (가장 최근 달 vs 그 이전 달 평균)
df_month = df.copy()
df_month["연월"] = df_month["날짜"].dt.to_period("M")
monthly_avg = df_month.groupby("연월", as_index=False)["가격"].mean()
monthly_avg = monthly_avg.sort_values("연월")

mom_delta = None
if len(monthly_avg) >= 2:
    latest = monthly_avg.iloc[-1]["가격"]
    previous = monthly_avg.iloc[-2]["가격"]
    mom_delta = (latest - previous) / previous * 100

# --- KPI ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("평균 가격", f"{avg_price:,.0f}원")
c2.metric("최고 가격", f"{max_price:,.0f}원", help=f"일자: {max_date:%Y-%m-%d}")
c3.metric("최저 가격", f"{min_price:,.0f}원", help=f"일자: {min_date:%Y-%m-%d}")
if mom_delta is not None:
    c4.metric("전월 대비 변화율", f"{mom_delta:+.2f}%")
else:
    c4.metric("전월 대비 변화율", "-")

st.divider()

# --- 가격 변화 그래프 ---
st.subheader(f"{selected_crop} 가격 변화")

view_mode = st.radio(
    "그래프 기준",
    options=["일별", "월별"],
    horizontal=True,
)

if view_mode == "일별":
    chart_df = df.copy()
    x_col = "날짜"
    title = f"{selected_crop} 일별 가격 변화"
else:
    chart_df = monthly_avg.copy()
    chart_df["연월"] = chart_df["연월"].astype(str)
    x_col = "연월"
    title = f"{selected_crop} 월별 평균 가격 변화"

fig = px.line(
    chart_df,
    x=x_col,
    y="가격",
    markers=True,
    title=title,
    labels={"가격": "가격 (원)", x_col: x_col},
)
fig.update_layout(yaxis_tickformat=",")
st.plotly_chart(fig, use_container_width=True)

st.divider()

# --- 5개 농산물 월별 등락률 비교 (3월~7월) ---
st.subheader("5개 농산물 월별 등락률")
st.caption("3월~7월 · 월평균 가격 기준 전월 대비 등락률(%) · 3월은 기준월(0%)")

mom_all = load_all_monthly_change_rates()
fig_mom = px.line(
    mom_all,
    x="연월",
    y="등락률",
    color="농산물",
    markers=True,
    title="농산물별 월별 등락률 비교 (2026년 3월~7월)",
    labels={"등락률": "등락률 (%)", "연월": "연월"},
    category_orders={"연월": MONTH_RANGE},
)
fig_mom.add_hline(y=0, line_dash="dash", line_color="gray")
fig_mom.update_layout(
    yaxis_ticksuffix="%",
    xaxis={"type": "category", "categoryorder": "array", "categoryarray": MONTH_RANGE},
)
st.plotly_chart(fig_mom, use_container_width=True)

st.divider()

# --- 선택 과제: 5개 농산물 평균 가격 비교 ---
st.subheader("농산물별 평균 가격 비교")
avg_all = load_all_average_prices()
fig_bar = px.bar(
    avg_all,
    x="농산물",
    y="평균가격",
    text="평균가격",
    title="5개 농산물 평균 가격 비교",
    labels={"평균가격": "평균 가격 (원)"},
    color="농산물",
)
fig_bar.update_traces(texttemplate="%{text:,.0f}원", textposition="outside")
fig_bar.update_layout(showlegend=False, yaxis_tickformat=",")
st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# --- 가격 데이터 표 ---
st.subheader("가격 데이터")
display_df = df.copy()
display_df["날짜"] = display_df["날짜"].dt.strftime("%Y-%m-%d")
display_df["가격"] = display_df["가격"].map(lambda x: f"{x:,.0f}")
st.dataframe(display_df, use_container_width=True, hide_index=True)

with st.expander("데이터 안내"):
    st.markdown(
        f"""
        - 출처: [KAMIS](https://www.kamis.or.kr/) 소매가격 기간별 자료
        - 선택 품목: **{selected_crop}** (`{CROP_FILES[selected_crop]}`)
        - 기간: {df['날짜'].min():%Y-%m-%d} ~ {df['날짜'].max():%Y-%m-%d}
        - 건수: {len(df):,}건
        """
    )
