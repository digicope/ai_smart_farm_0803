"""
지역별 식량작물 생산량 분석 대시보드 (실습 과제 답안)

실행 방법
--------
1) 이 파일이 있는 폴더로 이동
2) 필요 패키지 설치:
       pip install pandas plotly streamlit
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
    page_title="지역 식량작물 통계 대시보드",
    page_icon="🌾",
    layout="wide",
)

DATA_DIR = Path(__file__).resolve().parent
DATA_FILE = DATA_DIR / "식량작물_생산량_정곡__20260808172613.csv"

CROPS = ["미곡", "맥류", "잡곡", "두류", "서류"]
YEARS = list(range(2016, 2026))  # 2016~2025 (2026년 제외)


@st.cache_data
def load_production_data() -> pd.DataFrame:
    """KOSIS 식량작물 생산량(정곡) CSV를 장형식(long format)으로 정리합니다.

    원본 구조:
      - 0행: 연도
      - 1행: 작물:지표 (예: 미곡:생산량 (톤))
      - 2행~: 지역별 수치
      - 시도별(1)/시도별(2): 지역명 (소계이면 시도별(1), 아니면 시도별(2) 사용)
    """
    raw = pd.read_csv(DATA_FILE, encoding="cp949", header=None)

    year_row = raw.iloc[0, 2:]
    label_row = raw.iloc[1, 2:]

    records = []
    for row_idx in range(2, len(raw)):
        sido1 = str(raw.iloc[row_idx, 0]).strip()
        sido2 = str(raw.iloc[row_idx, 1]).strip()
        region = sido1 if sido2 == "소계" else sido2

        for col_idx in range(2, raw.shape[1]):
            year = int(year_row.iloc[col_idx - 2])
            if year not in YEARS:
                continue

            label = str(label_row.iloc[col_idx - 2])
            # 예: "미곡:생산량 (톤)" → crop=미곡, metric=생산량
            if ":" not in label:
                continue
            crop, metric_part = label.split(":", 1)
            if crop not in CROPS:
                continue
            if "생산량" not in metric_part:
                continue

            value = raw.iloc[row_idx, col_idx]
            if pd.isna(value) or str(value).strip() in {"-", ""}:
                production = None
            else:
                production = float(str(value).replace(",", ""))

            records.append(
                {
                    "지역": region,
                    "연도": year,
                    "작물": crop,
                    "생산량": production,
                }
            )

    df = pd.DataFrame(records)
    return df.sort_values(["지역", "작물", "연도"]).reset_index(drop=True)


def format_ton(value: float | None) -> str:
    """생산량(톤)을 천 단위 구분 문자열로 표시합니다."""
    if value is None or pd.isna(value):
        return "-"
    return f"{value:,.0f}톤"


# ------------------------------------------------------------
# 화면 구성
# ------------------------------------------------------------
st.title("지역 식량작물 통계 대시보드")
st.caption("KOSIS 식량작물 생산량(정곡) · 2016~2025년 · Streamlit + Pandas + Plotly")

df = load_production_data()

regions = df["지역"].drop_duplicates().tolist()
# 예시 화면과 같이 전라남도를 기본값으로
default_region = "전라남도" if "전라남도" in regions else regions[0]

col_a, col_b, col_c = st.columns(3)
with col_a:
    selected_region = st.selectbox("지역 선택", options=regions, index=regions.index(default_region))
with col_b:
    selected_crop = st.selectbox("작물 선택", options=CROPS, index=0)
with col_c:
    selected_year = st.selectbox("연도 선택", options=YEARS, index=len(YEARS) - 1)

# 선택 지역·작물의 연도별 시계열
region_crop_df = df[
    (df["지역"] == selected_region) & (df["작물"] == selected_crop)
].copy()
region_crop_df = region_crop_df.sort_values("연도")

# --- 핵심 통계 ---
year_row = region_crop_df.loc[region_crop_df["연도"] == selected_year, "생산량"]
selected_production = float(year_row.iloc[0]) if len(year_row) and pd.notna(year_row.iloc[0]) else None

valid = region_crop_df["생산량"].dropna()
avg_10y = float(valid.mean()) if len(valid) else None
max_10y = float(valid.max()) if len(valid) else None

# 선택 과제 1: 2016 → 2025 생산량 증감률
prod_2016_row = region_crop_df.loc[region_crop_df["연도"] == 2016, "생산량"]
prod_2025_row = region_crop_df.loc[region_crop_df["연도"] == 2025, "생산량"]
prod_2016 = float(prod_2016_row.iloc[0]) if len(prod_2016_row) and pd.notna(prod_2016_row.iloc[0]) else None
prod_2025 = float(prod_2025_row.iloc[0]) if len(prod_2025_row) and pd.notna(prod_2025_row.iloc[0]) else None

change_rate = None
if prod_2016 is not None and prod_2025 is not None and prod_2016 != 0:
    change_rate = (prod_2025 - prod_2016) / prod_2016 * 100

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{selected_year}년 생산량", format_ton(selected_production))
m2.metric("10년 평균", format_ton(avg_10y))
m3.metric("최대 생산량", format_ton(max_10y))
if change_rate is not None:
    m4.metric(
        "10년 증감률 (2016→2025)",
        f"{change_rate:+.2f}%",
        help=f"2016년 {format_ton(prod_2016)} → 2025년 {format_ton(prod_2025)}",
    )
else:
    m4.metric("10년 증감률 (2016→2025)", "-")

st.divider()

# --- 연도별 생산량 변화 (Line Chart) ---
st.subheader("연도별 생산량 변화")
line_df = region_crop_df.dropna(subset=["생산량"]).copy()
if line_df.empty:
    st.info(f"{selected_region} · {selected_crop} 생산량 데이터가 없습니다.")
else:
    fig_line = px.line(
        line_df,
        x="연도",
        y="생산량",
        markers=True,
        title=f"{selected_region} {selected_crop} 연도별 생산량 변화",
        labels={"생산량": "생산량 (톤)", "연도": "연도"},
    )
    fig_line.update_layout(
        xaxis=dict(dtick=1),
        yaxis_tickformat=",",
    )
    st.plotly_chart(fig_line, use_container_width=True)

st.divider()

# --- 지역별 생산량 비교 (Bar Chart) ---
st.subheader("지역별 생산량 비교")
compare_df = df[
    (df["작물"] == selected_crop)
    & (df["연도"] == selected_year)
    & (df["지역"] != "전국")
].copy()
compare_df = compare_df.dropna(subset=["생산량"]).sort_values("생산량", ascending=False)

if compare_df.empty:
    st.info(f"{selected_year}년 {selected_crop} 지역별 생산량 데이터가 없습니다.")
else:
    fig_bar = px.bar(
        compare_df,
        x="지역",
        y="생산량",
        title=f"{selected_year}년 {selected_crop} 생산량",
        labels={"생산량": "생산량 (톤)", "지역": "지역"},
        text="생산량",
    )
    fig_bar.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_bar.update_layout(yaxis_tickformat=",", xaxis_tickangle=-30)
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# --- 선택 과제 2: 식량작물별 비교 ---
st.subheader("식량작물별 비교")
st.caption(f"{selected_region} · {selected_year}년 기준 미곡/맥류/잡곡/두류/서류 생산량")
crop_compare_df = df[
    (df["지역"] == selected_region) & (df["연도"] == selected_year)
].copy()
crop_compare_df["작물"] = pd.Categorical(crop_compare_df["작물"], categories=CROPS, ordered=True)
crop_compare_df = crop_compare_df.sort_values("작물")
crop_compare_plot = crop_compare_df.dropna(subset=["생산량"])

if crop_compare_plot.empty:
    st.info(f"{selected_region} · {selected_year}년 작물별 생산량 데이터가 없습니다.")
else:
    fig_crop = px.bar(
        crop_compare_plot,
        x="작물",
        y="생산량",
        title=f"{selected_region} {selected_year}년 식량작물별 생산량",
        labels={"생산량": "생산량 (톤)", "작물": "작물"},
        text="생산량",
        color="작물",
        category_orders={"작물": CROPS},
    )
    fig_crop.update_traces(texttemplate="%{text:,.0f}", textposition="outside")
    fig_crop.update_layout(showlegend=False, yaxis_tickformat=",")
    st.plotly_chart(fig_crop, use_container_width=True)

st.divider()

# --- 선택 과제 3: 지역별 생산량 순위 ---
st.subheader("지역별 생산량 순위")
st.caption(f"{selected_year}년 {selected_crop} 생산량 기준 (전국 제외)")
rank_df = compare_df.copy()
if rank_df.empty:
    st.info(f"{selected_year}년 {selected_crop} 순위 데이터가 없습니다.")
else:
    rank_df = rank_df.reset_index(drop=True)
    rank_df.insert(0, "순위", range(1, len(rank_df) + 1))
    rank_df["순위"] = rank_df["순위"].map(lambda n: f"{n}위")
    rank_display = rank_df[["순위", "지역", "생산량"]].copy()
    rank_display["생산량"] = rank_display["생산량"].map(lambda x: f"{x:,.0f}톤")
    st.dataframe(rank_display, use_container_width=True, hide_index=True)

st.divider()

# --- 원본 데이터 ---
st.subheader("원본 데이터")
display_df = region_crop_df.copy()
display_df["생산량"] = display_df["생산량"].map(
    lambda x: f"{x:,.0f}" if pd.notna(x) else "-"
)
st.dataframe(display_df, use_container_width=True, hide_index=True)

with st.expander("데이터 안내"):
    st.markdown(
        f"""
        - 출처: [KOSIS](https://kosis.kr/) 식량작물 생산량(정곡)
        - 파일: `{DATA_FILE.name}`
        - 분석 기간: 2016~2025년
        - 선택: **{selected_region}** · **{selected_crop}** · **{selected_year}년**
        """
    )
