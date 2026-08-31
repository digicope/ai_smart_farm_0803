# -*- coding: utf-8 -*-
"""구어체 영농 메모를 창고 근거 기반 영농일지로 저장하는 Agent.

기상·농가·생육·병해충은 02번 SQLite 창고만 사용한다.
Open API serviceKey는 쓰지 않는다. OpenAI 키만 C:\\env\\.env 에서 로드한다.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

KST = ZoneInfo("Asia/Seoul")
BASE_DIR = Path(__file__).resolve().parent
COURSE_DIR = BASE_DIR.parent
DATA_DIR_02 = COURSE_DIR / "02_농업 데이터 수집 자동화 시스템 개발 실습" / "data"
WAREHOUSE_DIR = Path(r"C:\env\farm_warehouse")
DB_PATH = WAREHOUSE_DIR / "farm_warehouse.db"
ENV_PATH = r"C:\env\.env"

ALLOWED_CROPS = ("고추", "마늘", "밀", "배추", "사과", "양파", "옥수수", "콩", "포도", "논벼")
MISSING_CROPS = ("토마토", "딸기", "수박", "참외")
WORK_CATEGORIES = ("파종", "정식", "시비", "관수", "방제", "적심", "수확", "기타")
RAINFALL_HEAVY_MM = 20
RAINFALL_OUTLIER_MM = 400
HUMIDITY_HIGH_PCT = 85
TMAX_HOT_C = 33

CSV_TO_TABLE = {
    "weather": ("rda_weather_wanju_2023.csv", "weather"),
    "farm": ("rda_farm_info_2024.csv", "farm"),
    "growth": ("rda_growth_2024.csv", "growth"),
    "pest_info": ("rda_pest_catalog.csv", "pest_info"),
}

TOOL_LABELS = {
    "parse_work_memo": "메모 해석",
    "get_diary_weather": "당일 기상",
    "get_crop_field_context": "농가·생육",
    "search_pest_info": "병해충 검색",
    "save_farm_diary": "일지 저장",
    "query_saved_diaries": "일지 목록",
}

MEMO_A = (
    "2023년 8월 15일 완주 고추밭. 낮에 더워서 물 한번 줬고 "
    "웃거름도 조금 했어. 열매는 달렸는데 갯수는 세지 못함."
)
MEMO_B = (
    "8월 18일 완주 고추. 어제부터 비 오고 습해서 잎에 갈색 반점이 "
    "보임. 오후에 약 뿌렸음. 약 이름은 기억 안 남."
)
MEMO_C = (
    "2023-08-10 완주. 비가 너무 많이 와서 고추밭 물 빠지나 보고 "
    "고랑 정리만 했음. 수확은 안 함."
)
MEMO_D = (
    "2023년 8월 15일 전북 완주 양파밭은 이미 수확 끝난 뒤라 "
    "잔재 정리만 했어."
)
MEMO_E = (
    "오늘 토마토 하우스 정식하고 점적관수 시작했어. "
    "일지에 품종이랑 오늘 기온 넣어서 작성해줘."
)

SAMPLE_MEMOS = [
    {"id": "A", "title": "관수·시비, 병해 없음", "memo": MEMO_A},
    {"id": "B", "title": "방제 + 증상만, 약제명 없음", "memo": MEMO_B},
    {"id": "C", "title": "폭우 고랑 정리", "memo": MEMO_C},
    {"id": "D", "title": "완주 양파 잔재 정리", "memo": MEMO_D},
    {"id": "E", "title": "토마토(자료 없음)", "memo": MEMO_E},
]

SAMPLE_QUESTIONS = [
    {
        "label": "A. 관수·시비",
        "text": f"다음 메모를 영농일지로 작성하고 저장해줘.\n{MEMO_A}",
        "expected": [
            "parse_work_memo",
            "get_diary_weather",
            "get_crop_field_context",
            "save_farm_diary",
        ],
    },
    {
        "label": "B. 방제(약제명 없음)",
        "text": f"다음 메모를 영농일지로 작성하고 저장해줘.\n{MEMO_B}",
        "expected": [
            "parse_work_memo",
            "get_diary_weather",
            "search_pest_info",
            "save_farm_diary",
        ],
    },
    {
        "label": "C. 폭우 고랑 정리",
        "text": f"{MEMO_C} 일지로 남겨줘.",
        "expected": ["parse_work_memo", "get_diary_weather", "save_farm_diary"],
    },
    {
        "label": "D. 저장 목록",
        "text": "저장한 영농일지 중에서 고추, 2023년 8월 것만 목록으로 보여줘.",
        "expected": ["query_saved_diaries"],
    },
    {
        "label": "E. 토마토(자료 없음)",
        "text": MEMO_E,
        "expected": ["자료 없음 명시"],
    },
]

SYSTEM_PROMPT = """당신은 농가 구어체 메모를 구조화 영농일지로 바꾸는 Agent입니다.
기상·농가·생육·병해충 숫자는 도구가 돌려준 값만 사용하세요. 없는 값은 추측하지 마세요.

[데이터 범위]
- 농업기상 weather: 2023년, 전북 완주군 반교리·이서면
- 노지 농가 farm / 생육 growth: 2024년 참고 정보. 2023 작업일 기상으로 쓰지 마세요
- 있는 작목: 고추, 마늘, 밀, 배추, 사과, 양파, 옥수수, 콩, 포도, 논벼
- 없는 작목: 토마토, 딸기, 수박, 참외. 다른 작물로 대체하지 마세요
- 고추 생육은 전국 자료입니다. sido='전북'을 넣지 마세요

[도구 선택]
- 일지를 새로 작성할 때는 반드시 parse_work_memo를 먼저 호출하세요.
- parse 결과 work_date가 있으면 get_diary_weather를 호출하세요.
  work_date가 비었거나 lookup_weather=false 이면 get_diary_weather를 호출하지 마세요.
  '오늘'을 2023년의 아무 날짜로 바꿔 조회하지 마세요.
- 작물이 창고에 있는 작목(고추, 양파 등)이면 get_crop_field_context를 반드시 호출한 뒤 저장하세요.
- 병 증상·반점·약 살포가 있으면 search_pest_info.
- 근거를 모은 뒤 FarmDiary JSON을 만들어 save_farm_diary.
- 이미 저장한 일지 목록만 물으면 → query_saved_diaries. 새로 만들지 마세요
- 작업일이 비었거나 기상이 없으면 그 사실을 일지에 명시하세요. 오늘 날짜·오늘 기온을 지어내지 마세요
- 연도가 2023이 아닌 날짜의 기상은 조회해도 일지에 숫자를 채우지 마세요
- 토마토/딸기/수박/참외면 저장하지 않거나, 저장할 때 weather 숫자는 모두 null 이고 missing_reason을 적으세요.
  품종과 기온을 채우지 마세요. 고추로 바꾸지 마세요.

[약제·수량 안전]
- 메모에 없는 약제명·희석배수·살포량·수확량·품종을 채우지 마세요
- 방제만 있고 약 이름을 모르면 material="" 또는 "약제명 미기재", amount=""
- next_plan 에 구체 농약 제품명·희석배수를 쓰지 마세요
- 이서면 강수가 400mm를 넘으면 센서/원자료 이상입니다. 일지 rainfall_mm 에 넣지 마세요.
  반교리 값을 쓰세요

[save_farm_diary JSON 예]
{"work_date":"2023-08-15","region":"전북 완주군","crop":"고추","variety":"",
 "weather":{"tmin_c":24.0,"tmax_c":32.0,"tavg_c":27.12,"humidity_pct":85.83,
            "rainfall_mm":0.0,"station":"완주군 반교리","source_year":"2023","missing_reason":""},
 "work_items":[{"category":"관수","description":"물 한번 줌","material":"","amount":""}],
 "growth_note":"열매는 달렸으나 개수는 세지 못함(메모). 2024 생육표 수치로 단정하지 않음.",
 "pest_note":"","special_note":"","next_plan":"고온 지속 시 관수 상태 재확인",
 "evidence":{"weather_rows":true,"farm_rows":false,"growth_rows":true,"pest_names":[]},
 "raw_memo":"(원문)","narrative_ko":"(수치 포함 문장형 일지)"}

API 키를 출력하지 마세요.
save_farm_diary 결과에 warning 또는 weather.missing_reason이 있으면
기온 숫자를 답하지 말고 자료 부족 사유를 그대로 전달하세요.
"""


class WorkItem(BaseModel):
    model_config = ConfigDict(extra="ignore")
    category: Literal["파종", "정식", "시비", "관수", "방제", "적심", "수확", "기타"] = "기타"
    description: str = ""
    material: str = ""
    amount: str = ""

    @field_validator("category", mode="before")
    @classmethod
    def coerce_category(cls, value):
        if value in WORK_CATEGORIES:
            return value
        return "기타"


class WeatherBlock(BaseModel):
    model_config = ConfigDict(extra="ignore")
    tmin_c: float | None = None
    tmax_c: float | None = None
    tavg_c: float | None = None
    humidity_pct: float | None = None
    rainfall_mm: float | None = None
    station: str = ""
    source_year: str = ""
    missing_reason: str = ""


class Evidence(BaseModel):
    model_config = ConfigDict(extra="ignore")
    weather_rows: bool = False
    farm_rows: bool = False
    growth_rows: bool = False
    pest_names: list[str] = Field(default_factory=list)


class FarmDiary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    diary_id: int | None = None
    work_date: str = ""
    region: str = ""
    crop: str = ""
    variety: str = ""
    weather: WeatherBlock | None = None
    work_items: list[WorkItem] = Field(default_factory=list)
    growth_note: str = ""
    pest_note: str = ""
    special_note: str = ""
    next_plan: str = ""
    evidence: Evidence = Field(default_factory=Evidence)
    raw_memo: str = ""
    narrative_ko: str = ""


class ParsedMemo(BaseModel):
    model_config = ConfigDict(extra="ignore")
    work_date: str = Field(default="", description="YYYY-MM-DD. 없으면 빈 문자열")
    region: str = ""
    crop: str = ""
    variety: str = ""
    work_items: list[WorkItem] = Field(default_factory=list)
    symptoms: str = ""
    growth_observation: str = ""
    unclear: list[str] = Field(default_factory=list)
    pesticide_named: bool = False
    raw_memo: str = ""


PARSE_PROMPT = """영농 구어체 메모를 구조화하세요.
규칙:
- 오늘/어제처럼만 있고 달력이 없으면 work_date="" . 시스템 오늘 날짜를 넣지 마세요
- 연도 없이 8월 10일/15일/18일이면 2023-08-10/15/18
- 그 외 연도 없는 날짜는 work_date=""
- 작물을 단정할 수 없으면 crop=""
- 없는 작물(토마토, 딸기, 수박, 참외)도 메모에 있으면 그대로 crop에 적으세요. 고추로 바꾸지 마세요
- 약제 상품명·살포량이 없으면 pesticide_named=false, work_items.material과 amount는 빈 값
- 방제만 언급되면 category=방제, material=""
- variety는 메모에 있을 때만
- 작업 분류: 물/관수→관수, 웃거름/거름→시비, 약 살포→방제, 고랑·잔재 정리→기타.
  정식은 메모에 '정식'이 있을 때만. 물 주기나 웃거름을 정식으로 적지 마세요.
"""


def now_kst() -> str:
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M:%S")


def load_openai_key() -> str:
    load_dotenv(ENV_PATH)
    key = os.getenv("OPENAI_API_KEY") or ""
    if not key:
        raise ValueError(f"OPENAI_API_KEY 를 {ENV_PATH}에서 찾을 수 없습니다.")
    return key


def connect_db() -> sqlite3.Connection:
    WAREHOUSE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (table,),
    ).fetchone()
    return bool(row)


def table_count(conn: sqlite3.Connection, table: str) -> int:
    if not table_exists(conn, table):
        return 0
    try:
        row = conn.execute(f"SELECT COUNT(*) AS n FROM {table}").fetchone()
        return int(row["n"]) if row else 0
    except sqlite3.Error:
        return 0


def date_span(conn: sqlite3.Connection, table: str, col: str) -> str:
    if not table_exists(conn, table):
        return "테이블 없음"
    try:
        row = conn.execute(f"SELECT MIN({col}) AS a, MAX({col}) AS b FROM {table}").fetchone()
        if not row or row["a"] is None:
            return "없음"
        return f"{row['a']} ~ {row['b']}"
    except sqlite3.Error:
        return "없음"


def ingest_csv_if_needed(source: str) -> str:
    filename, table = CSV_TO_TABLE[source]
    path = DATA_DIR_02 / filename
    if not path.exists():
        return f"{source}: 02 폴더 CSV 없음 ({path})"
    conn = connect_db()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df.to_sql(table, conn, if_exists="replace", index=False)
        conn.commit()
        return f"{source} 적재: {filename} → {table} ({len(df)}건)"
    finally:
        conn.close()


def ensure_warehouse() -> list[str]:
    notes = []
    conn = connect_db()
    try:
        missing = []
        for source, (_fn, table) in CSV_TO_TABLE.items():
            if table_count(conn, table) == 0:
                missing.append(source)
    finally:
        conn.close()
    for source in missing:
        notes.append(ingest_csv_if_needed(source))
    if not notes:
        notes.append("창고 테이블 weather/farm/growth/pest_info 가 이미 있습니다.")
    return notes


def init_diary_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS farm_diary (
            diary_id INTEGER PRIMARY KEY AUTOINCREMENT,
            work_date TEXT,
            region TEXT,
            crop TEXT,
            variety TEXT,
            weather_json TEXT,
            work_items_json TEXT,
            growth_note TEXT,
            pest_note TEXT,
            special_note TEXT,
            next_plan TEXT,
            evidence_json TEXT,
            raw_memo TEXT,
            narrative_ko TEXT,
            diary_json TEXT,
            saved_at TEXT
        )
        """
    )
    conn.commit()


def reset_diary_table() -> str:
    conn = connect_db()
    try:
        conn.execute("DROP TABLE IF EXISTS farm_diary")
        init_diary_schema(conn)
        return "farm_diary 테이블을 새로 만들었습니다."
    finally:
        conn.close()


def warehouse_overview() -> pd.DataFrame:
    ensure_warehouse()
    conn = connect_db()
    try:
        rows = []
        specs = [
            ("weather", "obs_date", "2023 완주 기상"),
            ("farm", "plant_date", "2024 노지 농가"),
            ("growth", "survey_date", "2024 노지 생육"),
            ("pest_info", "", "병해충 목록"),
        ]
        for table, col, note in specs:
            rows.append(
                {
                    "table": table,
                    "건수": table_count(conn, table),
                    "기간": date_span(conn, table, col) if col else "-",
                    "설명": note,
                }
            )
        return pd.DataFrame(rows)
    finally:
        conn.close()


def sample_weather_rows() -> pd.DataFrame:
    ensure_warehouse()
    conn = connect_db()
    try:
        df = pd.read_sql_query(
            """
            SELECT station, obs_date, tmin_c, tmax_c, tavg_c, humidity_pct, rainfall_mm
            FROM weather
            WHERE obs_date IN ('2023-08-10','2023-08-15','2023-08-18')
            ORDER BY obs_date, station
            """,
            conn,
        )
        return df
    finally:
        conn.close()


def normalize_work_date(raw: str, memo: str) -> str:
    text = (raw or "").strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        return text
    m = re.search(r"(20\d{2})\s*[년.\-/]\s*(\d{1,2})\s*[월.\-/]\s*(\d{1,2})", memo)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", memo)
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    if re.search(r"오늘|금일", memo) and not re.search(r"20\d{2}|\d{1,2}월", memo):
        return ""
    m = re.search(r"(?<!\d)(\d{1,2})\s*월\s*(\d{1,2})\s*일", memo)
    if m:
        month, day = int(m.group(1)), int(m.group(2))
        if (month, day) in {(8, 10), (8, 15), (8, 18)}:
            return f"2023-{month:02d}-{day:02d}"
        return ""
    return ""


def guess_crop(memo: str, parsed_crop: str) -> str:
    crop = (parsed_crop or "").strip()
    if crop:
        return crop
    for name in list(MISSING_CROPS) + list(ALLOWED_CROPS):
        if name in memo:
            return name
    return ""


def memo_supports_date(memo: str, work_date: str) -> bool:
    if not work_date or not memo:
        return False
    if work_date in memo:
        return True
    parts = work_date.split("-")
    if len(parts) != 3:
        return False
    year, month, day = parts[0], int(parts[1]), int(parts[2])
    compact = re.sub(r"\s+", "", memo)
    candidates = [
        f"{year}년{month}월{day}일",
        f"{month}월{day}일",
        f"{year}-{month:02d}-{day:02d}",
        f"{year}/{month}/{day}",
    ]
    return any(item in compact for item in candidates)


def refine_work_items(items: list[dict], memo: str) -> list[dict]:
    text = memo or ""
    inferred: list[dict] = []

    def add(category: str, description: str, material: str = "") -> None:
        inferred.append(
            {"category": category, "description": description, "material": material, "amount": ""}
        )

    if re.search(r"정식", text) and not re.search(r"잔재|고랑", text):
        add("정식", "정식")
    if re.search(r"점적관수|물\s*(한번|한 번|줌|줬)|관수", text):
        add("관수", "관수")
    if re.search(r"웃거름|거름|시비", text):
        add("시비", "웃거름" if "웃거름" in text else "시비")
    if re.search(r"약\s*뿌|방제", text):
        unnamed = bool(re.search(r"기억 안|약 이름|이름은 기억", text))
        add("방제", "방제", "약제명 미기재" if unnamed else "")
    if re.search(r"고랑|잔재", text):
        add("기타", "고랑/잔재 정리")
    if inferred:
        return inferred
    return items


def parse_memo_text(memo: str) -> dict:
    load_openai_key()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured = llm.with_structured_output(ParsedMemo)
    parsed: ParsedMemo = structured.invoke(
        [
            {"role": "system", "content": PARSE_PROMPT},
            {"role": "user", "content": memo},
        ]
    )
    parsed.raw_memo = memo
    parsed.work_date = normalize_work_date(parsed.work_date, memo)
    parsed.crop = guess_crop(memo, parsed.crop)
    parsed.work_items = [WorkItem.model_validate(x) for x in refine_work_items(
        [item.model_dump() for item in parsed.work_items], memo
    )]
    if parsed.crop in MISSING_CROPS:
        parsed.unclear = list(dict.fromkeys([*parsed.unclear, f"창고에 없는 작물:{parsed.crop}"]))
    if not parsed.pesticide_named:
        for item in parsed.work_items:
            if item.category == "방제":
                if "기억" in memo or "약 이름" in memo:
                    item.material = "약제명 미기재"
                item.amount = ""
    payload = parsed.model_dump()
    lookup = bool(parsed.work_date) and parsed.crop not in MISSING_CROPS
    payload["lookup_weather"] = lookup
    suggested: list[str] = []
    if lookup:
        suggested.append("get_diary_weather")
    else:
        suggested.append("get_diary_weather 호출 금지(날짜 없음 또는 없는 작물)")
    if parsed.crop in MISSING_CROPS:
        suggested.append("없는 작물: 기온·품종 창작 금지, 고추로 대체 금지")
    elif parsed.crop in ALLOWED_CROPS:
        suggested.append("get_crop_field_context")
    if parsed.symptoms or any(i.category == "방제" for i in parsed.work_items):
        suggested.append("search_pest_info")
    payload["suggested_next_tools"] = suggested
    return payload


def weather_flags(rainfall: float | None, humidity: float | None, tmax: float | None) -> list[str]:
    flags = []
    if rainfall is not None and rainfall >= RAINFALL_OUTLIER_MM:
        flags.append("센서/원자료 이상 가능(400mm 초과). 일지 강수량으로 쓰지 말 것")
    elif rainfall is not None and rainfall >= RAINFALL_HEAVY_MM:
        flags.append("폭우(20mm 이상)")
    if humidity is not None and humidity >= HUMIDITY_HIGH_PCT:
        flags.append("다습(85% 이상)")
    if tmax is not None and tmax >= TMAX_HOT_C:
        flags.append("고온(33℃ 이상)")
    return flags


def row_weather_dict(row: dict) -> dict:
    def num(key: str) -> float | None:
        val = row.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return None

    rainfall = num("rainfall_mm")
    humidity = num("humidity_pct")
    tmax = num("tmax_c")
    return {
        "station": str(row.get("station") or ""),
        "tmin_c": num("tmin_c"),
        "tmax_c": tmax,
        "tavg_c": num("tavg_c"),
        "humidity_pct": humidity,
        "rainfall_mm": rainfall,
        "source_year": "2023",
        "flags": weather_flags(rainfall, humidity, tmax),
        "use_as_diary_rainfall": not (rainfall is not None and rainfall >= RAINFALL_OUTLIER_MM),
    }


def get_weather(work_date: str, prefer_station: str = "반교리") -> dict:
    date = (work_date or "").strip()
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        return {
            "found": False,
            "reason": "작업일(YYYY-MM-DD)이 없습니다. 오늘 날짜로 대체하지 마세요.",
        }
    year = date[:4]
    if year != "2023":
        return {
            "found": False,
            "work_date": date,
            "reason": f"창고 기상은 2023년뿐입니다. {year}년 날짜를 2023으로 매핑하지 마세요.",
        }
    conn = connect_db()
    try:
        if table_count(conn, "weather") == 0:
            return {"found": False, "reason": "weather 테이블이 비어 있습니다."}
        rows = conn.execute(
            """
            SELECT station, obs_date, tmin_c, tmax_c, tavg_c, humidity_pct, rainfall_mm
            FROM weather
            WHERE obs_date = ? AND sigungu LIKE '%완주%'
            ORDER BY station
            """,
            (date,),
        ).fetchall()
        if not rows:
            return {
                "found": False,
                "work_date": date,
                "reason": "해당 날짜 기상이 창고에 없습니다. 인접 날짜로 대체하지 마세요.",
            }
        items = [row_weather_dict(dict(r)) for r in rows]
        primary = None
        refs = []
        for item in items:
            if prefer_station and prefer_station in item["station"]:
                primary = item
            else:
                refs.append(item)
        if primary is None:
            primary = items[0]
            refs = items[1:]
        return {
            "found": True,
            "work_date": date,
            "primary": primary,
            "reference": refs,
            "instruction": "일지 weather 블록에는 primary(반교리 우선) 숫자만 넣으세요. "
            "reference의 400mm 초과 강수는 쓰지 마세요.",
        }
    finally:
        conn.close()


def crop_missing_message(crop: str) -> str | None:
    name = (crop or "").strip()
    if name in MISSING_CROPS:
        return (
            f"내려받은 노지 CSV에 '{name}' 자료가 없습니다. "
            f"가능한 작목: {', '.join(ALLOWED_CROPS)}. 다른 작물로 대체하지 마세요."
        )
    return None


def get_field_context(crop: str, sido: str = "", sigungu: str = "") -> dict:
    name = (crop or "").strip()
    missing = crop_missing_message(name)
    if missing:
        return {
            "crop": name,
            "in_warehouse": False,
            "message": missing,
            "farm_rows": 0,
            "growth_rows": 0,
        }
    if not name:
        return {"crop": "", "message": "작물이 비어 있습니다.", "farm_rows": 0, "growth_rows": 0}

    conn = connect_db()
    try:
        farm_sql = "SELECT sido, sigungu, farm_id, crop, variety, area_m2, plant_date, harvest_date, yield_total, note FROM farm WHERE crop LIKE ?"
        farm_params: list = [f"%{name}%"]
        if sido:
            farm_sql += " AND sido LIKE ?"
            farm_params.append(f"%{sido}%")
        if sigungu:
            farm_sql += " AND sigungu LIKE ?"
            farm_params.append(f"%{sigungu}%")
        farm_sql += " LIMIT 8"
        farm_df = pd.read_sql_query(farm_sql, conn, params=farm_params)

        growth_note = ""
        growth_sql = "SELECT sido, sigungu, crop, survey_date, plant_height_cm, fruit_count FROM growth WHERE crop LIKE ?"
        growth_params: list = [f"%{name}%"]
        if name == "고추":
            growth_note = "고추 생육은 전국 자료입니다. 전북 필터를 적용하지 않았습니다."
        else:
            if sido:
                growth_sql += " AND sido LIKE ?"
                growth_params.append(f"%{sido}%")
            if sigungu:
                growth_sql += " AND sigungu LIKE ?"
                growth_params.append(f"%{sigungu}%")
        growth_all = pd.read_sql_query(growth_sql, conn, params=growth_params)
        growth_summary = {}
        if len(growth_all):
            growth_summary = {
                "rows": int(len(growth_all)),
                "survey_from": str(growth_all["survey_date"].min()),
                "survey_to": str(growth_all["survey_date"].max()),
                "mean_height_cm": round(float(pd.to_numeric(growth_all["plant_height_cm"], errors="coerce").mean()), 1)
                if "plant_height_cm" in growth_all.columns
                else None,
                "note": "2024년 참고 생육입니다. 2023 작업일 기상·당일 열매 수로 쓰지 마세요.",
            }
        return {
            "crop": name,
            "in_warehouse": True,
            "year_note": "farm/growth 는 2024년 참고 정보",
            "growth_filter_note": growth_note,
            "farm_rows": int(len(farm_df)),
            "farm_sample": farm_df.head(5).to_dict(orient="records"),
            "growth_rows": int(len(growth_all)),
            "growth_summary": growth_summary,
        }
    finally:
        conn.close()


def strip_pest_detail(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text or "")
    text = re.sub(r"imageList:.*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:280]


def search_pests(crop: str, symptom: str) -> dict:
    crop_q = (crop or "").strip()
    symptom_q = (symptom or "").strip()
    if not crop_q and not symptom_q:
        return {"hits": [], "message": "작물 또는 증상이 필요합니다."}
    conn = connect_db()
    try:
        if table_count(conn, "pest_info") == 0:
            return {"hits": [], "message": "pest_info 테이블이 비어 있습니다."}
        df = pd.read_sql_query("SELECT pest_name, detail FROM pest_info", conn)
    finally:
        conn.close()

    tokens = [t for t in re.split(r"[\s,./]+", symptom_q) if len(t) >= 2]
    if "반점" in symptom_q and "점무늬" not in tokens:
        tokens.append("점무늬")

    scored = []
    for _, row in df.iterrows():
        name = str(row.get("pest_name") or "")
        detail = str(row.get("detail") or "")
        blob = name + " " + detail
        score = 0
        if crop_q and crop_q in blob:
            score += 10
        for tok in tokens:
            if tok and tok in blob:
                score += 4
            if tok and tok in name:
                score += 3
        if score <= 0:
            continue
        scored.append((score, name, strip_pest_detail(detail)))
    scored.sort(key=lambda x: (-x[0], x[1]))
    uniq = []
    seen = set()
    for score, name, detail in scored:
        key = (name, detail[:40])
        if key in seen:
            continue
        seen.add(key)
        uniq.append({"pest_name": name, "detail": detail, "score": score})
        if len(uniq) >= 3:
            break
    return {
        "crop": crop_q,
        "symptom": symptom_q,
        "hits": uniq,
        "caution": "검색 결과에 없는 방제 약제명·살포량을 만들지 마세요. 병명을 단정하지 못하면 '자료에서 특정하지 못함'이라고 적으세요.",
    }


def extract_json_object(text: str) -> dict:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("JSON 객체를 찾지 못했습니다.")
    return json.loads(raw[start : end + 1])


def empty_weather(reason: str) -> WeatherBlock:
    return WeatherBlock(missing_reason=reason)


def sanitize_diary(diary: FarmDiary) -> FarmDiary:
    memo = diary.raw_memo or ""
    if diary.crop in MISSING_CROPS:
        diary.variety = ""
        reason = (
            f"창고에 '{diary.crop}' 농가·생육이 없습니다. "
            "다른 작물로 대체하지 않았고, 품종·오늘 기온을 채우지 않았습니다."
        )
        if "오늘" in memo or not memo_supports_date(memo, diary.work_date):
            diary.work_date = ""
            reason = (
                f"작업일이 '오늘'이거나 메모에 달력 날짜가 없어 창고(2023) 기상과 맞출 수 없습니다. "
                f"'{diary.crop}'는 노지 CSV에 없습니다. 기온·품종을 지어내지 않았습니다."
            )
        diary.weather = empty_weather(reason)
        diary.special_note = (diary.special_note + " " + reason).strip()
        if not diary.growth_note:
            diary.growth_note = "자료 없음"
        if not diary.narrative_ko or "기온" in diary.narrative_ko:
            diary.narrative_ko = (
                f"{diary.crop} 작업 메모는 확인했으나, 창고에 해당 작물 자료가 없고 "
                "오늘 기온을 창고에서 대조할 수 없어 일지 기상·품종은 비워 두었습니다."
            )
        return diary

    if diary.work_date and not memo_supports_date(memo, diary.work_date) and "오늘" in memo:
        diary.work_date = ""
        diary.weather = empty_weather("메모에 달력 날짜가 없어 기상을 채우지 않았습니다.")
        return diary

    if diary.weather and diary.weather.rainfall_mm is not None:
        if diary.weather.rainfall_mm >= RAINFALL_OUTLIER_MM:
            diary.weather.rainfall_mm = None
            extra = "이서면 등 400mm 초과 강수는 일지 강수량에서 제외했습니다."
            diary.special_note = (diary.special_note + " " + extra).strip()

    if diary.work_date:
        warehouse = get_weather(diary.work_date)
        if warehouse.get("found") and warehouse.get("primary"):
            primary = warehouse["primary"]
            rain = primary["rainfall_mm"] if primary.get("use_as_diary_rainfall") else None
            diary.weather = WeatherBlock(
                tmin_c=primary.get("tmin_c"),
                tmax_c=primary.get("tmax_c"),
                tavg_c=primary.get("tavg_c"),
                humidity_pct=primary.get("humidity_pct"),
                rainfall_mm=rain,
                station=primary.get("station") or "",
                source_year="2023",
                missing_reason="" if rain is not None or primary.get("rainfall_mm") is None else "강수 이상치로 강수량만 비움",
            )
        elif diary.weather and (diary.weather.tmin_c is not None or diary.weather.tmax_c is not None):
            reason = warehouse.get("reason") or "해당 날짜 기상이 창고에 없습니다."
            diary.weather = empty_weather(reason)

    for item in diary.work_items:
        if item.category == "방제" and ("약 이름" in memo or "기억 안" in memo):
            item.material = "약제명 미기재"
            item.amount = ""
    return diary


def save_diary(diary_json: str) -> dict:
    try:
        payload = extract_json_object(diary_json)
    except (ValueError, json.JSONDecodeError) as exc:
        return {"ok": False, "error": f"JSON 파싱 실패: {exc}"}
    payload.pop("diary_id", None)
    try:
        diary = FarmDiary.model_validate(payload)
    except ValidationError as exc:
        return {"ok": False, "error": f"FarmDiary 검증 실패: {exc}"}
    diary = sanitize_diary(diary)
    conn = connect_db()
    try:
        init_diary_schema(conn)
        dumped = diary.model_dump()
        cur = conn.execute(
            """
            INSERT INTO farm_diary(
                work_date, region, crop, variety, weather_json, work_items_json,
                growth_note, pest_note, special_note, next_plan, evidence_json,
                raw_memo, narrative_ko, diary_json, saved_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                diary.work_date,
                diary.region,
                diary.crop,
                diary.variety,
                json.dumps(dumped.get("weather"), ensure_ascii=False),
                json.dumps(dumped.get("work_items"), ensure_ascii=False),
                diary.growth_note,
                diary.pest_note,
                diary.special_note,
                diary.next_plan,
                json.dumps(dumped.get("evidence"), ensure_ascii=False),
                diary.raw_memo,
                diary.narrative_ko,
                json.dumps(dumped, ensure_ascii=False),
                now_kst(),
            ),
        )
        conn.commit()
        diary_id = int(cur.lastrowid)
        warning = ""
        if diary.weather and diary.weather.missing_reason:
            warning = diary.weather.missing_reason
        return {
            "ok": True,
            "diary_id": diary_id,
            "work_date": diary.work_date,
            "crop": diary.crop,
            "weather": diary.weather.model_dump() if diary.weather else None,
            "warning": warning,
            "saved_at": now_kst(),
            "instruction": "warning이 있으면 기온·품종을 사용자에게 숫자로 말하지 말고 자료 부족을 전하세요.",
        }
    finally:
        conn.close()


def query_diaries(crop: str = "", start_date: str = "", end_date: str = "") -> dict:
    conn = connect_db()
    try:
        init_diary_schema(conn)
        sql = """
            SELECT diary_id, work_date, crop, region, work_items_json, saved_at
            FROM farm_diary WHERE 1=1
        """
        params: list = []
        if crop:
            sql += " AND crop LIKE ?"
            params.append(f"%{crop}%")
        if start_date:
            sql += " AND work_date >= ?"
            params.append(start_date)
        if end_date:
            sql += " AND work_date <= ?"
            params.append(end_date)
        sql += " ORDER BY work_date, diary_id"
        rows = conn.execute(sql, params).fetchall()
        items = []
        for row in rows:
            try:
                works = json.loads(row["work_items_json"] or "[]")
                summary = ", ".join(
                    f"{w.get('category', '')}:{w.get('description', '')}" for w in works[:3]
                )
            except json.JSONDecodeError:
                summary = ""
            items.append(
                {
                    "diary_id": row["diary_id"],
                    "work_date": row["work_date"],
                    "crop": row["crop"],
                    "region": row["region"],
                    "work_summary": summary[:120],
                    "saved_at": row["saved_at"],
                }
            )
        return {"count": len(items), "items": items}
    finally:
        conn.close()


def load_diary_table() -> pd.DataFrame:
    conn = connect_db()
    try:
        init_diary_schema(conn)
        return pd.read_sql_query(
            """
            SELECT diary_id, work_date, region, crop, variety,
                   weather_json, work_items_json, growth_note, pest_note,
                   special_note, next_plan, narrative_ko, saved_at
            FROM farm_diary
            ORDER BY diary_id
            """,
            conn,
        )
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
    ensure_warehouse()
    conn = connect_db()
    init_diary_schema(conn)
    conn.close()

    @tool
    def parse_work_memo(memo: str) -> str:
        """구어체 영농 메모에서 날짜, 지역, 작물, 작업, 자재, 증상을 JSON으로 뽑습니다.
        작업일이 없으면 오늘 날짜를 넣지 않습니다. lookup_weather=false이면 기상 조회를 하지 마세요.
        """
        try:
            return json.dumps(parse_memo_text(memo), ensure_ascii=False)
        except Exception as exc:
            return f"메모 해석 실패: {type(exc).__name__}: {exc}"

    @tool
    def get_diary_weather(work_date: str, prefer_station: str = "반교리") -> str:
        """완주 2023 농업기상에서 작업일 하루를 조회합니다.
        반교리 행을 우선하고 이서면은 참고입니다. 없는 날짜는 인접일로 대체하지 않습니다.
        2023년이 아니면 매핑하지 않습니다.
        """
        try:
            return json.dumps(get_weather(work_date, prefer_station), ensure_ascii=False)
        except Exception as exc:
            return f"기상 조회 실패: {type(exc).__name__}: {exc}"

    @tool
    def get_crop_field_context(crop: str, sido: str = "", sigungu: str = "") -> str:
        """2024 노지 농가·생육 참고 정보를 요약합니다.
        고추 생육은 전국 자료이므로 sido에 전북을 넣지 마세요.
        토마토, 딸기, 수박, 참외는 창고에 없습니다.
        """
        try:
            return json.dumps(get_field_context(crop, sido, sigungu), ensure_ascii=False)
        except Exception as exc:
            return f"농가·생육 조회 실패: {type(exc).__name__}: {exc}"

    @tool
    def search_pest_info(crop: str, symptom: str) -> str:
        """병해충 목록에서 작물·증상 키워드로 상위 3건을 검색합니다.
        원문에 없는 방제 약제는 넣지 않습니다.
        """
        try:
            return json.dumps(search_pests(crop, symptom), ensure_ascii=False)
        except Exception as exc:
            return f"병해충 검색 실패: {type(exc).__name__}: {exc}"

    @tool
    def save_farm_diary(diary_json: str) -> str:
        """FarmDiary JSON을 검증한 뒤 farm_diary 테이블에 저장합니다.
        검증 실패 시 저장하지 않습니다. 성공 시 diary_id를 반환합니다.
        """
        try:
            return json.dumps(save_diary(diary_json), ensure_ascii=False)
        except Exception as exc:
            return f"일지 저장 실패: {type(exc).__name__}: {exc}"

    @tool
    def query_saved_diaries(crop: str = "", start_date: str = "", end_date: str = "") -> str:
        """저장된 영농일지 목록을 돌려줍니다. 본문 전체가 아니라 id·날짜·작물·작업 요약·저장시각입니다.
        목록만 물을 때는 새 일지를 만들지 마세요.
        """
        try:
            return json.dumps(query_diaries(crop, start_date, end_date), ensure_ascii=False)
        except Exception as exc:
            return f"일지 목록 조회 실패: {type(exc).__name__}: {exc}"

    tools = [
        parse_work_memo,
        get_diary_weather,
        get_crop_field_context,
        search_pest_info,
        save_farm_diary,
        query_saved_diaries,
    ]
    agent = create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)
    return {
        "agent": agent,
        "tools": [t.name for t in tools],
        "db_path": str(DB_PATH),
        "data_dir": str(DATA_DIR_02),
    }


def ask_agent(runtime: dict, question: str) -> dict:
    result = runtime["agent"].invoke(
        {"messages": [{"role": "user", "content": question}]},
        config={"recursion_limit": 40},
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
