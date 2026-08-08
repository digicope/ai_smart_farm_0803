"""
지역별 식량작물 생산량 분석 대시보드 (실습 과제 답안)

제공 CSV(2016~2025)를 읽어 지역·작물·연도별 생산량을 분석합니다.
실행: streamlit run app.py
중지: Ctrl + C
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="지역 식량작물 통계 대시보드",
    page_icon="🌾",
    layout="centered",
)

DATA_DIR = Path(__file__).resolve().parent
CSV_CANDIDATES = [
    "식량작물_생산량_정곡.csv",
    "식량작물_생산량_정곡__20260808172613.csv",
]
CROPS = ["미곡", "맥류", "잡곡", "두류", "서류"]
YEARS = list(range(2016, 2026))  # 2016~2025 (2026 제외)


def _to_number(value) -> float | None:
    """'-', 빈 값, 쉼표 포함 숫자를 float로 변환합니다."""
    if pd.isna(value):
        return None
    text = str(value).strip().replace(",", "")
    if text in {"", "-", "nan", "None"}:
        return None
    return float(text)


def _find_csv_path() -> Path:
    for name in CSV_CANDIDATES:
        path = DATA_DIR / name
        if path.exists():
            return path
    parent = DATA_DIR.parent
    for name in CSV_CANDIDATES:
        path = parent / name
        if path.exists():
            return path
    raise FileNotFoundError(
        "CSV 파일이 없습니다. "
        "식량작물_생산량_정곡.csv 또는 "
        "식량작물_생산량_정곡__20260808172613.csv 를 같은 폴더에 두세요."
    )


@st.cache_data
def load_production_data() -> pd.DataFrame:
    """통계청형 다중 헤더 CSV를 장형(long) 데이터로 변환합니다.

    원본 구조:
    - 0행: 연도
    - 1행: 지표명 (미곡:생산량 (톤) 등)
    - 2행~: 지역별 값
    """
    path = _find_csv_path()
    raw = pd.read_csv(path, encoding="cp949", header=None)

    years = raw.iloc[0, 2:].astype(str).str.strip().tolist()
    metrics = raw.iloc[1, 2:].astype(str).str.strip().tolist()
    body = raw.iloc[2:].reset_index(drop=True)

    records = []
    for i in range(len(body)):
        sido1 = str(body.iloc[i, 0]).strip()
        sido2 = str(body.iloc[i, 1]).strip()
        # 전남광주통합특별시처럼 하위 지역이 있으면 시도별(2) 사용
        region = sido2 if sido2 != "소계" else sido1

        for j, (year_text, metric) in enumerate(zip(years, metrics)):
            year = int(float(year_text))
            if year not in YEARS:
                continue

            crop = None
            for name in CROPS:
                if metric.startswith(f"{name}:") and "생산량" in metric:
                    crop = name
                    break
            if crop is None:
                continue

            records.append(
                {
                    "지역": region,
                    "연도": year,
                    "작물": crop,
                    "생산량": _to_number(body.iloc[i, j + 2]),
                }
            )

    df = pd.DataFrame(records)
    return df.sort_values(["지역", "작물", "연도"]).reset_index(drop=True)


def format_ton(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.0f}톤"


# ------------------------------------------------------------
# 화면 구성
# ------------------------------------------------------------
st.title("지역 식량작물 통계 대시보드")
st.caption("2016~2025년 지역별 식량작물 생산량 | Pandas + Plotly + Streamlit")

try:
    df = load_production_data()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

if df.empty:
    st.warning("분석할 생산량 데이터가 없습니다.")
    st.stop()

regions = [r for r in df["지역"].unique().tolist() if r != "전국"]
default_region = "전라남도" if "전라남도" in regions else regions[0]

col_a, col_b, col_c = st.columns(3)
with col_a:
    region = st.selectbox("지역 선택", regions, index=regions.index(default_region))
with col_b:
    crop = st.selectbox("작물 선택", CROPS)
with col_c:
    year = st.selectbox("연도 선택", YEARS, index=len(YEARS) - 1)

region_crop = df[(df["지역"] == region) & (df["작물"] == crop)].copy()
region_crop_valid = region_crop.dropna(subset=["생산량"])

selected_row = region_crop.loc[region_crop["연도"] == year, "생산량"]
selected_prod = float(selected_row.iloc[0]) if not selected_row.empty else None
avg_10y = float(region_crop_valid["생산량"].mean()) if not region_crop_valid.empty else None
max_10y = float(region_crop_valid["생산량"].max()) if not region_crop_valid.empty else None

# 선택 과제 1: 2016→2025 증감률
prod_2016 = region_crop.loc[region_crop["연도"] == 2016, "생산량"]
prod_2025 = region_crop.loc[region_crop["연도"] == 2025, "생산량"]
v2016 = float(prod_2016.iloc[0]) if not prod_2016.empty and pd.notna(prod_2016.iloc[0]) else None
v2025 = float(prod_2025.iloc[0]) if not prod_2025.empty and pd.notna(prod_2025.iloc[0]) else None
if v2016 and v2016 != 0 and v2025 is not None:
    change_rate = (v2025 - v2016) / v2016 * 100
else:
    change_rate = None

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{year}년 생산량", format_ton(selected_prod))
m2.metric("10년 평균", format_ton(avg_10y))
m3.metric("최대 생산량", format_ton(max_10y))
m4.metric(
    "10년 증감률",
    f"{change_rate:+.1f}%" if change_rate is not None else "-",
)

st.divider()

# (3) 연도별 생산량 변화 - Line Chart
st.subheader("연도별 생산량 변화")
if region_crop_valid.empty:
    st.info(f"{region} · {crop} 생산량 데이터가 없습니다.")
else:
    fig_line = px.line(
        region_crop_valid,
        x="연도",
        y="생산량",
        markers=True,
        title=f"{region} {crop} 연도별 생산량 변화",
    )
    fig_line.update_layout(xaxis_title="연도", yaxis_title="생산량(톤)")
    fig_line.update_xaxes(dtick=1)
    st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# (4) 지역별 생산량 비교 - Bar Chart
st.subheader("지역별 생산량 비교")
year_crop = df[(df["연도"] == year) & (df["작물"] == crop) & (df["지역"] != "전국")].copy()
year_crop = year_crop.dropna(subset=["생산량"]).sort_values("생산량", ascending=False)

if year_crop.empty:
    st.info(f"{year}년 {crop} 지역별 생산량 데이터가 없습니다. (일부 작물은 2025년 미공표)")
else:
    fig_bar = px.bar(
        year_crop,
        x="생산량",
        y="지역",
        orientation="h",
        text_auto=".0f",
        title=f"{year}년 {crop} 생산량",
    )
    fig_bar.update_layout(
        xaxis_title="생산량(톤)",
        yaxis_title="지역",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # 선택 과제 3: 지역별 생산량 순위
    st.markdown(f"**{year}년 {crop} 지역별 생산량 순위**")
    rank_df = year_crop.reset_index(drop=True).copy()
    rank_df.insert(0, "순위", range(1, len(rank_df) + 1))
    rank_df["생산량"] = rank_df["생산량"].map(lambda v: f"{v:,.0f}")
    st.dataframe(rank_df[["순위", "지역", "생산량"]], use_container_width=True, hide_index=True)

st.divider()

# 선택 과제 2: 식량작물별 비교
st.subheader(f"{region} 식량작물별 생산량 비교 ({year}년)")
crop_cmp = df[(df["지역"] == region) & (df["연도"] == year)].copy()
crop_cmp = crop_cmp.dropna(subset=["생산량"])
crop_cmp["작물"] = pd.Categorical(crop_cmp["작물"], categories=CROPS, ordered=True)
crop_cmp = crop_cmp.sort_values("작물")

if crop_cmp.empty:
    st.info(f"{region} · {year}년 작물별 생산량 데이터가 없습니다.")
else:
    fig_crop = px.bar(
        crop_cmp,
        x="작물",
        y="생산량",
        text_auto=".0f",
        title=f"{region} {year}년 식량작물별 생산량",
        color="작물",
    )
    fig_crop.update_layout(xaxis_title="작물", yaxis_title="생산량(톤)", showlegend=False)
    st.plotly_chart(fig_crop, use_container_width=True)

st.divider()

st.subheader("원본 데이터")
display_df = region_crop.copy()
display_df["생산량"] = display_df["생산량"].map(
    lambda v: f"{v:,.0f}" if pd.notna(v) else "-"
)
st.dataframe(display_df, use_container_width=True, hide_index=True)
