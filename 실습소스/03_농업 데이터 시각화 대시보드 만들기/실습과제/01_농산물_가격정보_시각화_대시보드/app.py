"""
농산물 가격 정보 시각화 대시보드 (실습 과제 답안)

같은 폴더에 제공된 KAMIS Excel 파일을 읽어 대시보드를 구성합니다.
- 딸기_가격정보.xlsx
- 배추_가격정보.xlsx
- 사과_가격정보.xlsx
- 수박_가격정보.xlsx
- 쌀_가격정보.xlsx

실행: streamlit run app.py
중지: Ctrl + C
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="농산물 가격 정보 대시보드",
    page_icon="🥬",
    layout="centered",
)

DATA_DIR = Path(__file__).resolve().parent

PRODUCT_FILES = {
    "딸기": "딸기_가격정보.xlsx",
    "배추": "배추_가격정보.xlsx",
    "사과": "사과_가격정보.xlsx",
    "수박": "수박_가격정보.xlsx",
    "쌀": "쌀_가격정보.xlsx",
}


def _to_number(series: pd.Series) -> pd.Series:
    """'5,214', '-' 같은 값을 숫자로 변환합니다."""
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .replace({"-": None, "nan": None, "None": None})
    )
    return pd.to_numeric(cleaned, errors="coerce")


@st.cache_data
def load_price_data(product: str) -> pd.DataFrame:
    """제공된 Excel에서 일자별 가격 행만 읽어 정리합니다.

    KAMIS 파일 구조:
    - 0행: 제목
    - 1행: 컬럼명 (날짜, 가격, 등락률)
    - 2~4행: 평년/전년/당월 요약
    - 5행~: YYYY.MM.DD 일자 데이터
    """
    file_path = DATA_DIR / PRODUCT_FILES[product]
    if not file_path.exists():
        raise FileNotFoundError(f"엑셀 파일이 없습니다: {file_path.name}")

    raw = pd.read_excel(file_path, header=None)
    df = raw.iloc[2:].copy()
    df.columns = ["날짜", "가격", "등락률"]

    # 실제 일자 행만 사용 (평년 월, 전년 월, 당월 평균 제외)
    date_mask = df["날짜"].astype(str).str.match(r"^\d{4}\.\d{2}\.\d{2}$", na=False)
    df = df.loc[date_mask].copy()

    df["날짜"] = pd.to_datetime(df["날짜"], format="%Y.%m.%d")
    df["가격"] = _to_number(df["가격"])
    df["등락률"] = _to_number(df["등락률"])
    df = df.dropna(subset=["가격"]).sort_values("날짜").reset_index(drop=True)
    return df


@st.cache_data
def load_all_avg_prices() -> pd.DataFrame:
    """5개 농산물의 평균 가격을 모아 비교용 DataFrame을 만듭니다."""
    rows = []
    for product in PRODUCT_FILES:
        df = load_price_data(product)
        rows.append({"농산물": product, "평균가격": df["가격"].mean()})
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 화면 구성
# ------------------------------------------------------------
st.title("농산물 가격 정보 대시보드")
st.caption("제공 Excel 데이터(KAMIS) | Pandas + Plotly + Streamlit")

product = st.selectbox("농산물 선택", list(PRODUCT_FILES.keys()))

try:
    df = load_price_data(product)
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if df.empty:
    st.warning("선택한 농산물의 일자별 가격 데이터가 없습니다.")
    st.stop()

avg_price = df["가격"].mean()
max_price = df["가격"].max()
min_price = df["가격"].min()

# 전월 대비: 월별 평균 가격의 최근 두 달 비교
monthly = (
    df.assign(연월=df["날짜"].dt.to_period("M").astype(str))
    .groupby("연월", as_index=False)["가격"]
    .mean()
    .sort_values("연월")
)
if len(monthly) >= 2:
    prev = monthly.iloc[-2]["가격"]
    curr = monthly.iloc[-1]["가격"]
    change_rate = (curr - prev) / prev * 100 if prev else 0.0
else:
    change_rate = 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("평균 가격", f"{avg_price:,.0f}원")
c2.metric("최고 가격", f"{max_price:,.0f}원")
c3.metric("최저 가격", f"{min_price:,.0f}원")
c4.metric("전월 대비", f"{change_rate:+.1f}%")

st.divider()

st.subheader(f"{product} 가격 변화")
fig = px.line(
    df,
    x="날짜",
    y="가격",
    markers=True,
    title=f"{product} 가격 변화",
)
fig.update_layout(xaxis_title="날짜", yaxis_title="가격(원)")
st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("가격 데이터")
display_df = df.copy()
display_df["날짜"] = display_df["날짜"].dt.strftime("%Y-%m-%d")
st.dataframe(display_df, use_container_width=True)

st.divider()
st.subheader("농산물별 평균 가격 비교")
avg_df = load_all_avg_prices()
fig_bar = px.bar(
    avg_df,
    x="농산물",
    y="평균가격",
    text_auto=".0f",
    title="5개 농산물 평균 가격 비교",
)
fig_bar.update_layout(yaxis_title="평균 가격(원)")
st.plotly_chart(fig_bar, use_container_width=True)
