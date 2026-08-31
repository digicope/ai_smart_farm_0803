# -*- coding: utf-8 -*-
"""공공데이터포털 CSV를 Agent Tool로 내려받아 SQLite 창고에 적재한다.

파일 다운로드(fileDownload.do)는 Tool이 호출한다. Open API serviceKey는 쓰지 않는다.
OpenAI 키만 C:\\env\\.env 에서 로드한다.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from portal_csv import download_and_prepare

KST = ZoneInfo("Asia/Seoul")
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
WAREHOUSE_DIR = Path(r"C:\env\farm_warehouse")
DB_PATH = WAREHOUSE_DIR / "farm_warehouse.db"
ENV_PATH = r"C:\env\.env"
COLLECTION_CONFIG = {
    "title": "농촌진흥청·공공데이터포털 CSV 수집",
    "region_focus": {
        "sido": "전북",
        "sigungu": "완주군",
        "note": "전주 인근. 농업기상은 완주군 반교리·이서면 관측점(2023).",
    },
    "sources": [
        {
            "source_id": "weather",
            "name": "병해충발생예측 활용 농업기상",
            "file": "rda_weather_wanju_2023.csv",
            "table": "weather",
            "year": "2023",
            "provider": "농촌진흥청",
            "portal": "https://www.data.go.kr/data/15136768/fileData.do",
            "description": "국가농작물병해충관리 기상정보 중 완주군 일별 집계",
        },
        {
            "source_id": "farm",
            "name": "노지 현장 농가정보",
            "file": "rda_farm_info_2024.csv",
            "table": "farm",
            "year": "2024",
            "provider": "농촌진흥청",
            "portal": "https://www.data.go.kr/data/15126334/fileData.do",
            "description": "노지 농가 작목·품종·정식·수확량",
        },
        {
            "source_id": "growth",
            "name": "노지 작물 생육조사",
            "file": "rda_growth_2024.csv",
            "table": "growth",
            "year": "2024",
            "provider": "농촌진흥청",
            "portal": "https://www.data.go.kr/data/15126334/fileData.do",
            "description": "고추(전국)와 전북 밀·배추·양파·콩 생육",
        },
        {
            "source_id": "pest_sites",
            "name": "병해충 예찰조사 지점",
            "file": "rda_pest_sites.csv",
            "table": "pest_sites",
            "year": "2015-2024",
            "provider": "농촌진흥청",
            "portal": "https://www.data.go.kr/data/15123424/fileData.do",
            "description": "작목별 예찰포 위치·면적",
        },
        {
            "source_id": "pest_info",
            "name": "병해충 목록",
            "file": "rda_pest_catalog.csv",
            "table": "pest_info",
            "year": "2025",
            "provider": "농림수산식품교육문화정보원",
            "portal": "https://www.data.go.kr/data/15151253/fileData.do",
            "description": "NCPMS·PSIS 병해충명과 상세 요약",
        },
    ],
    "alert_thresholds": {
        "humidity_high_pct": 85,
        "rainfall_heavy_mm": 20,
        "rainfall_sensor_outlier_mm": 400,
        "tmax_hot_c": 33,
    },
    "crops_in_data": ["고추", "마늘", "밀", "배추", "사과", "양파", "옥수수", "콩", "포도", "논벼"],
    "crops_not_in_data": ["토마토", "딸기", "수박", "참외"],
}

SOURCE_FILES = {
    "weather": DATA_DIR / "rda_weather_wanju_2023.csv",
    "farm": DATA_DIR / "rda_farm_info_2024.csv",
    "growth": DATA_DIR / "rda_growth_2024.csv",
    "pest_sites": DATA_DIR / "rda_pest_sites.csv",
    "pest_info": DATA_DIR / "rda_pest_catalog.csv",
}
TABLE_BY_SOURCE = {
    "weather": "weather",
    "farm": "farm",
    "growth": "growth",
    "pest_sites": "pest_sites",
    "pest_info": "pest_info",
}
DATE_COL = {
    "weather": "obs_date",
    "farm": "plant_date",
    "growth": "survey_date",
    "pest_sites": "year",
    "pest_info": "",
}
ALLOWED_CROPS = ("고추", "마늘", "밀", "배추", "사과", "양파", "옥수수", "콩", "포도", "논벼")
MISSING_CROPS = ("토마토", "딸기", "수박", "참외")
TOOL_LABELS = {
    "list_data_sources": "수집대상 목록",
    "download_portal_data": "포털 CSV 다운로드",
    "ingest_farm_data": "CSV 적재",
    "get_collection_status": "수집 현황",
    "query_collected_data": "창고 조회",
    "detect_farm_alerts": "이상 경보",
}
SYSTEM_PROMPT = """당신은 농촌진흥청·공공데이터포털 파일데이터를 수집하는 농업 데이터 Agent입니다.
Open API 인증키(serviceKey)는 쓰지 않습니다. CSV 파일 다운로드는 download_portal_data Tool이 수행합니다.
Tool이 돌려준 숫자·날짜·작물만 사용하세요. 없는 값은 추측하지 마세요.

[데이터 범위]
- 농업기상: 2023년, 전북 완주군 반교리·이서면 (전주 인근)
- 노지 농가·생육: 2024년. 작목은 고추, 마늘, 밀, 배추, 사과, 양파, 옥수수, 콩, 포도
- 예찰 지점: 논벼 등, 시도·시군 필터 가능
- 병해충 목록: 병해충명 검색
- 토마토, 딸기, 수박, 참외 생육/농가는 이 CSV에 없습니다. 다른 작물로 대체하지 마세요.

[도구 선택]
- 수집 가능한 종류만 물으면 → list_data_sources. 다운로드하지 마세요.
- 내려받아/수집해/포털에서 받아 → download_portal_data. 전부면 source='all'
  이어서 창고에 넣으려면 ingest_farm_data 도 호출하세요.
- 이미 받은 CSV만 적재 → ingest_farm_data. 파일이 없으면 download_portal_data 를 먼저 호출
- 적재 건수·기간 → get_collection_status
- 수치·명단 조회 → query_collected_data
  table: weather, farm, growth, pest_sites, pest_info
  전북/완주/김제 등은 sido, sigungu 로 필터
  병해충명은 table='pest_info', crop 자리에 병명(예: 역병)
- 이상/점검/경보 → detect_farm_alerts 를 반드시 호출
- 조회 테이블이 비었으면 download 후 ingest
- 고추 생육은 전국 자료입니다. 기상만 완주로 필터하세요. 고추 조회에 sido='전북'을 넣으면 0건입니다.

[답변]
- 출처(농촌진흥청/공공데이터포털)와 연도를 밝히세요
- 수치에 단위와 관측점/지역을 붙이세요
- API 키나 파일 절대경로를 장황하게 출력하지 마세요
"""

SAMPLE_QUESTIONS = [
    {
        "label": "A. 수집 대상",
        "text": "공공데이터포털에서 받아 자동 수집할 수 있는 농업 데이터 종류를 알려줘.",
    },
    {
        "label": "B. 전체 수집",
        "text": "공공데이터포털에서 농업 CSV를 내려받고 창고에 적재한 뒤 수집 현황을 보고해줘.",
    },
    {
        "label": "C. 완주 기상과 고추 생육",
        "text": "2023년 8월 완주군 농업기상(강수·습도·기온)과 2024년 고추 생육(전국)을 짧게 정리해줘.",
    },
    {
        "label": "D. 이상 점검",
        "text": "전북 완주 인근 기상과 노지 농가 자료를 기준으로 이상·병해 관련 사항을 점검해줘.",
    },
    {
        "label": "E. 토마토(자료 없음)",
        "text": "토마토 하우스 생육 데이터와 정식 후 물 관리 기록을 보여줘.",
    },
]


def load_openai_key() -> str:
    load_dotenv(ENV_PATH)
    key = os.getenv("OPENAI_API_KEY") or ""
    if not key:
        raise ValueError(f"OPENAI_API_KEY 를 {ENV_PATH}에서 찾을 수 없습니다.")
    return key


def load_config() -> dict:
    """포털 URL·임계값·작목 목록. data/ JSON이 아니라 코드의 COLLECTION_CONFIG를 쓴다."""
    return COLLECTION_CONFIG


def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def connect_db() -> sqlite3.Connection:
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_log (
            source_id TEXT PRIMARY KEY,
            file_name TEXT,
            row_count INTEGER,
            ingested_at TEXT
        )
        """
    )
    conn.commit()


def ingest_source(source: str) -> str:
    source = (source or "").strip().lower()
    if source == "all":
        return "\n".join(ingest_source(name) for name in SOURCE_FILES)
    if source not in SOURCE_FILES:
        return f"알 수 없는 source='{source}'. 가능: {', '.join(SOURCE_FILES)}, all"
    path = SOURCE_FILES[source]
    if not path.exists():
        return f"파일이 없습니다: {path.name}. download_portal_data(source='{source}') 를 먼저 호출하세요."
    conn = connect_db()
    try:
        init_schema(conn)
        df = pd.read_csv(path, encoding="utf-8-sig")
        table = TABLE_BY_SOURCE[source]
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.execute(
            """
            INSERT INTO ingest_log(source_id, file_name, row_count, ingested_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(source_id) DO UPDATE SET
                file_name=excluded.file_name,
                row_count=excluded.row_count,
                ingested_at=excluded.ingested_at
            """,
            (source, path.name, int(len(df)), now_kst()),
        )
        conn.commit()
        return f"{source} 적재 완료: {path.name} → {table} ({len(df)}건)"
    finally:
        conn.close()


def file_preview_count(path: Path) -> int:
    if not path.exists():
        return 0
    return int(len(pd.read_csv(path, encoding="utf-8-sig")))


def format_sources() -> str:
    config = load_config()
    region = config["region_focus"]
    lines = [
        f"수집 초점: {region['sido']} {region['sigungu']} ({region['note']})",
        "제공: 농촌진흥청·공공데이터포털 파일데이터(CSV). Open API 인증키 없음.",
        f"있는 작목: {', '.join(config['crops_in_data'])}",
        f"없는 작목: {', '.join(config['crops_not_in_data'])}",
        "",
        "[수집 대상 CSV]",
    ]
    for item in config["sources"]:
        path = DATA_DIR / item["file"]
        lines.append(
            f"- {item['source_id']}: {item['name']} / {item['file']} / "
            f"{item['year']} / 원본 {file_preview_count(path)}건 / {item['description']}"
        )
        lines.append(f"  포털: {item['portal']}")
    return "\n".join(lines)


def table_count(conn: sqlite3.Connection, table: str) -> int:
    try:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        return int(row["n"]) if row else 0
    except sqlite3.Error:
        return 0


def date_span(conn: sqlite3.Connection, table: str, col: str) -> str:
    if not col:
        return "-"
    try:
        row = conn.execute(f"SELECT MIN({col}) AS a, MAX({col}) AS b FROM {table}").fetchone()
        if not row or row["a"] is None:
            return "없음"
        return f"{row['a']} ~ {row['b']}"
    except sqlite3.Error:
        return "없음"


def format_status() -> str:
    conn = connect_db()
    try:
        init_schema(conn)
        lines = [f"창고: {DB_PATH.name}"]
        for source, table in TABLE_BY_SOURCE.items():
            n = table_count(conn, table)
            span = date_span(conn, table, DATE_COL.get(table, ""))
            lines.append(f"{table} {n}건 기간 {span}")
        lines.append("")
        lines.append("[적재 로그]")
        logs = conn.execute("SELECT * FROM ingest_log ORDER BY source_id").fetchall()
        if not logs:
            lines.append("아직 적재한 출처가 없습니다. ingest_farm_data 를 먼저 호출하세요.")
        else:
            for row in logs:
                lines.append(
                    f"- {row['source_id']}: {row['file_name']} {row['row_count']}건 @ {row['ingested_at']}"
                )
        return "\n".join(lines)
    finally:
        conn.close()


def query_table(
    table: str,
    sido: str = "",
    sigungu: str = "",
    crop: str = "",
    start_date: str = "",
    end_date: str = "",
) -> str:
    key = (table or "").strip().lower()
    if key not in TABLE_BY_SOURCE.values() and key not in TABLE_BY_SOURCE:
        return f"알 수 없는 table='{table}'. 가능: weather, farm, growth, pest_sites, pest_info"
    real_table = TABLE_BY_SOURCE.get(key, key)
    source_id = next(s for s, t in TABLE_BY_SOURCE.items() if t == real_table)
    conn = connect_db()
    try:
        init_schema(conn)
        if table_count(conn, real_table) == 0:
            return f"{real_table} 가 비어 있습니다. ingest_farm_data(source='{source_id}') 를 먼저 호출하세요."

        crop_q = (crop or "").strip()
        if crop_q in MISSING_CROPS:
            return (
                f"내려받은 노지 CSV에 '{crop_q}' 자료가 없습니다. "
                f"가능한 작목: {', '.join(ALLOWED_CROPS)}. 다른 작물 데이터로 대체하지 마세요."
            )

        cols = [r[1] for r in conn.execute(f"PRAGMA table_info({real_table})").fetchall()]
        sql = f"SELECT * FROM {real_table} WHERE 1=1"
        params: list = []
        if sido and "sido" in cols:
            sql += " AND sido LIKE ?"
            params.append(f"%{sido}%")
        if sigungu and "sigungu" in cols:
            sql += " AND sigungu LIKE ?"
            params.append(f"%{sigungu}%")
        if crop_q and real_table == "pest_info" and "pest_name" in cols:
            sql += " AND (pest_name LIKE ? OR detail LIKE ?)"
            params.extend([f"%{crop_q}%", f"%{crop_q}%"])
        elif crop_q and "crop" in cols:
            sql += " AND crop LIKE ?"
            params.append(f"%{crop_q}%")
        date_col = DATE_COL.get(real_table, "")
        if start_date and date_col and date_col in cols:
            sql += f" AND {date_col} >= ?"
            params.append(start_date)
        if end_date and date_col and date_col in cols:
            sql += f" AND {date_col} <= ?"
            params.append(end_date)
        if date_col and date_col in cols:
            sql += f" ORDER BY {date_col}"
        sql += " LIMIT 80"
        rows = conn.execute(sql, params).fetchall()
        if not rows:
            hint = date_span(conn, real_table, date_col) if date_col else "-"
            return (
                f"{real_table} 조회 0건 "
                f"(sido={sido or '-'}, sigungu={sigungu or '-'}, crop={crop_q or '-'}, "
                f"{start_date or '-'}~{end_date or '-'}). 실제 기간: {hint}"
            )
        df = pd.DataFrame([dict(r) for r in rows])
        extra = ""
        if real_table == "growth" and "plant_height_cm" in df.columns:
            extra = (
                f"\n요약: {len(df)}건(최대 80건 표시) "
                f"초장 평균 {pd.to_numeric(df['plant_height_cm'], errors='coerce').mean():.1f}cm"
            )
        if real_table == "weather" and "rainfall_mm" in df.columns:
            rain = pd.to_numeric(df["rainfall_mm"], errors="coerce")
            sane = rain[rain < 400]
            extra = (
                f"\n요약: 강수합(400mm미만) {sane.sum():.1f}mm, "
                f"최고기온 {pd.to_numeric(df['tmax_c'], errors='coerce').max():.1f}℃, "
                f"평균습도 {pd.to_numeric(df['humidity_pct'], errors='coerce').mean():.1f}%"
            )
        return f"{real_table} {len(df)}건{extra}\n{df.head(40).to_csv(index=False)}"
    finally:
        conn.close()


def detect_alerts() -> str:
    config = load_config()
    th = config["alert_thresholds"]
    conn = connect_db()
    try:
        init_schema(conn)
        missing = [n for n in ("weather", "farm") if table_count(conn, n) == 0]
        if missing:
            return (
                f"경보에 필요한 테이블이 비어 있습니다: {', '.join(missing)}. "
                "ingest_farm_data(source='all') 또는 먼저 download_portal_data 를 호출하세요."
            )
        weather = pd.read_sql_query("SELECT * FROM weather ORDER BY obs_date", conn)
        farm = pd.read_sql_query("SELECT * FROM farm", conn)
        alerts: list[str] = []

        wet = weather[pd.to_numeric(weather["humidity_pct"], errors="coerce") >= th["humidity_high_pct"]]
        if len(wet):
            sample = ", ".join(sorted(wet["obs_date"].astype(str).unique())[:8])
            alerts.append(
                f"[고습] 임계 {th['humidity_high_pct']}% 이상 {wet['obs_date'].nunique()}일 "
                f"(최고 {wet['humidity_pct'].max()}%, 예: {sample})"
            )

        rain = weather[pd.to_numeric(weather["rainfall_mm"], errors="coerce") >= th["rainfall_heavy_mm"]]
        outlier = rain[pd.to_numeric(rain["rainfall_mm"], errors="coerce") >= th["rainfall_sensor_outlier_mm"]]
        rain_ok = rain[pd.to_numeric(rain["rainfall_mm"], errors="coerce") < th["rainfall_sensor_outlier_mm"]]
        for _, row in rain_ok.nlargest(5, "rainfall_mm").iterrows():
            alerts.append(
                f"[강우] {row['obs_date']} {row['station']} {row['rainfall_mm']}mm "
                f"(임계 {th['rainfall_heavy_mm']}mm)"
            )
        for _, row in outlier.iterrows():
            alerts.append(
                f"[강수 관측 이상] {row['obs_date']} {row['station']} 일합계 {row['rainfall_mm']}mm "
                f"(원자료 시간값 합산, {th['rainfall_sensor_outlier_mm']}mm 초과 → 센서 오류 가능)"
            )

        hot = weather[pd.to_numeric(weather["tmax_c"], errors="coerce") >= th["tmax_hot_c"]]
        if len(hot):
            alerts.append(
                f"[고온] 최고기온 {th['tmax_hot_c']}℃ 이상 {hot['obs_date'].nunique()}일 "
                f"(최고 {hot['tmax_c'].max()}℃)"
            )

        notes = farm[farm["note"].fillna("").str.contains("병|역병|무름|폭염|수확포기")]
        for _, row in notes.iterrows():
            alerts.append(
                f"[농가 비고] {row['sido']} {row['sigungu']} {row['crop']} "
                f"농가 {row['farm_id']}: {row['note']}"
            )

        if table_count(conn, "pest_sites"):
            sites = pd.read_sql_query(
                "SELECT sido, crop, COUNT(*) AS n FROM pest_sites "
                "WHERE sido LIKE '%전북%' AND CAST(year AS TEXT) LIKE '2024%' "
                "GROUP BY sido, crop",
                conn,
            )
            for _, row in sites.iterrows():
                alerts.append(f"[예찰지점] 2024 {row['sido']} {row['crop']} {int(row['n'])}개소")

        header = (
            f"임계값: 습도>={th['humidity_high_pct']}%, 강수>={th['rainfall_heavy_mm']}mm, "
            f"최고기온>={th['tmax_hot_c']}℃ / 기상 연도 2023 완주, 농가 연도 2024"
        )
        if not alerts:
            return header + "\n설정된 임계값을 넘는 항목이 없습니다."
        return header + "\n" + "\n".join(alerts)
    finally:
        conn.close()


def message_text(msg) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("text"):
                parts.append(str(block["text"]))
        return "\n".join(parts)
    return str(content)


def extract_tool_trace(result: dict) -> list[dict]:
    traces: list[dict] = []
    for msg in result.get("messages", []):
        for call in getattr(msg, "tool_calls", None) or []:
            name = call.get("name") if isinstance(call, dict) else getattr(call, "name", "")
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", {})
            traces.append({"name": name, "label": TOOL_LABELS.get(name, name), "args": args or {}})
        if type(msg).__name__ == "ToolMessage":
            preview = message_text(msg).replace("\n", " ")
            traces.append(
                {
                    "name": getattr(msg, "name", "tool"),
                    "label": "결과",
                    "preview": preview[:280],
                }
            )
    return traces


def build_runtime() -> dict:
    load_openai_key()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    conn = connect_db()
    init_schema(conn)
    conn.close()

    @tool
    def list_data_sources() -> str:
        """공공데이터포털에서 Agent가 내려받을 수 있는 농업 CSV 목록을 보여줍니다.
        Open API 인증키는 없습니다. 목록만 필요할 때는 이 도구만 쓰고 다운로드하지 마세요.
        """
        try:
            return format_sources()
        except Exception as exc:
            return f"수집 대상 조회 실패: {type(exc).__name__}: {exc}"

    @tool
    def download_portal_data(source: str = "all", force: bool = False) -> str:
        """공공데이터포털 파일데이터 URL에서 CSV/ZIP을 다운로드하고 실습용 CSV로 정리합니다.
        source: weather, farm, growth, pest_sites, pest_info, all.
        Open API serviceKey는 쓰지 않습니다. 기상 원본(약 85MB)은 정리본이 있으면 생략합니다.
        다시 받으려면 force=True.
        """
        try:
            return download_and_prepare(source, force=force)
        except Exception as exc:
            return f"포털 다운로드 실패: {type(exc).__name__}: {exc}"

    @tool
    def ingest_farm_data(source: str = "all") -> str:
        """내려받은 CSV를 SQLite 창고에 적재합니다.
        source: weather, farm, growth, pest_sites, pest_info, all.
        파일이 없으면 download_portal_data 를 먼저 호출하세요.
        """
        try:
            return ingest_source(source)
        except Exception as exc:
            return f"적재 실패: {type(exc).__name__}: {exc}"

    @tool
    def get_collection_status() -> str:
        """창고 적재 현황(테이블별 건수, 기간, 적재 시각)을 반환합니다."""
        try:
            return format_status()
        except Exception as exc:
            return f"현황 조회 실패: {type(exc).__name__}: {exc}"

    @tool
    def query_collected_data(
        table: str,
        sido: str = "",
        sigungu: str = "",
        crop: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> str:
        """적재된 CSV를 조회합니다.
        table: weather, farm, growth, pest_sites, pest_info.
        sido 예: 전북, sigungu 예: 완주/김제/진안.
        crop 예: 고추, 밀, 배추. pest_info 에서는 병명(역병, 진딧물).
        기상 날짜는 2023-MM-DD, 생육은 2024-MM-DD.
        """
        try:
            return query_table(table, sido, sigungu, crop, start_date, end_date)
        except Exception as exc:
            return f"조회 실패: {type(exc).__name__}: {exc}"

    @tool
    def detect_farm_alerts() -> str:
        """완주 2023 기상 임계값과 2024 농가 비고(병해·폭염)로 경보를 만듭니다.
        원자료에 없는 경보는 만들지 않습니다.
        """
        try:
            return detect_alerts()
        except Exception as exc:
            return f"경보 탐지 실패: {type(exc).__name__}: {exc}"

    agent = create_agent(
        model=llm,
        tools=[
            list_data_sources,
            download_portal_data,
            ingest_farm_data,
            get_collection_status,
            query_collected_data,
            detect_farm_alerts,
        ],
        system_prompt=SYSTEM_PROMPT,
    )
    cfg = load_config()
    return {
        "agent": agent,
        "db_path": str(DB_PATH),
        "data_dir": str(DATA_DIR),
        "region": cfg["region_focus"]["sigungu"],
        "crops": ALLOWED_CROPS,
    }


def ask_agent(runtime: dict, question: str) -> dict:
    result = runtime["agent"].invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 30},
    )
    called = []
    for item in extract_tool_trace(result):
        if item.get("label") != "결과" and item.get("name"):
            called.append(item["name"])
    return {
        "answer": message_text(result["messages"][-1]),
        "tools": called,
        "trace": extract_tool_trace(result),
    }
