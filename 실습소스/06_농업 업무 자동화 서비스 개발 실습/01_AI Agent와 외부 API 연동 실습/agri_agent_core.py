# -*- coding: utf-8 -*-
"""기상청 단기·중기예보 + 농업기술길잡이 PDF Agent 코어."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import unquote
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

KST = ZoneInfo("Asia/Seoul")
DATA_DIR = Path(__file__).resolve().parent
VS_DIR = Path(r"C:\env\crop_guide_faiss")
ENV_PATH = r"C:\env\.env"

LOCATION_NAME = "전주"
NX, NY = 63, 89
LAND_REG_ID = "11F10000"
TA_REG_ID = "11F10201"
STN_ID = "146"

SHORT_TERM_BASE = "https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
MID_TERM_BASE = "https://apis.data.go.kr/1360000/MidFcstInfoService"

ALLOWED_CROPS = ("토마토", "고추", "수박", "참외", "고구마", "사과")
CROP_KEYWORDS = [
    ("토마토", "토마토"),
    ("고추", "고추"),
    ("수박", "수박"),
    ("참외", "참외"),
    ("고구마", "고구마"),
    ("사과", "사과"),
]
LAND_TO_STN = {
    "11B00000": "109",
    "11F10000": "146",
    "11F20000": "156",
    "11C20000": "133",
    "11C10000": "131",
    "11H10000": "143",
    "11H20000": "159",
    "11G00000": "184",
    "11D10000": "105",
}
SKY_CODE = {"1": "맑음", "3": "구름많음", "4": "흐림"}
PTY_CODE = {
    "0": "없음",
    "1": "비",
    "2": "비/눈",
    "3": "눈",
    "4": "소나기",
    "5": "빗방울",
    "6": "빗방울눈날림",
    "7": "눈날림",
}
TOOL_LABELS = {
    "get_short_term_weather": "단기예보",
    "get_mid_term_weather": "중기예보",
    "search_crop_guide": "작물 길잡이 검색",
}
SYSTEM_PROMPT = """당신은 농업 의사결정을 돕는 Agent입니다.
날씨는 기상청 API Tool 결과만, 재배법은 농업기술길잡이 PDF 검색 결과만 근거로 답합니다.
없는 숫자·약제명·작업법을 추측하지 마세요. 근거가 없으면 '주어진 자료만으로는 알 수 없다'고 말하세요.

[도구 선택 — 반드시 지키세요]
- 오늘/내일/모레, 1~3일 시간별 상세 → get_short_term_weather 만
- 4일 이후 전망만 물으면 → get_mid_term_weather
- '앞으로 10일간 날씨를 분석해서 농작업 계획'처럼 1~10일을 모두 다루면
  반드시 get_short_term_weather 와 get_mid_term_weather 를 둘 다 호출한 뒤
  search_crop_guide 도 호출하세요. 중기만으로 10일 계획을 끝내지 마세요.
  중기예보의 +3일은 비어 있을 수 있습니다. 오늘~모레는 단기예보가 필요합니다.
- 재배/병해충/정식/시비/수확/육묘 지식 → search_crop_guide
  농작업 계획이면 검색어를 구체화하세요. 예: '정식 주의사항', '강우 시 방제', '관수'
- 작물이 없는 단순 날씨 질문에는 PDF를 호출하지 마세요
- 날씨가 필요 없는 재배 질문에는 기상청 API를 호출하지 마세요
- PDF에 없는 작물(딸기, 벼 등)도 search_crop_guide(crop='딸기')를 한 번 호출해
  자료 없음을 확인하세요. 다른 작물로 다시 검색하지 마세요

[작물]
- 검색 가능한 작물: 토마토, 고추, 수박, 참외, 고구마, 사과

[답변 형식]
- 날씨 답에는 실제 예보 수치(기온, 강수확률 등)를 넣으세요
- PDF를 썼으면 파일명과 페이지를 반드시 밝히세요
- 10일 농작업 계획은 D+0~2(단기 상세)와 D+3~10(중기 전망)으로 나눠 쓰세요
- 인증키, 격자 좌표, 구역코드를 사용자에게 출력하지 마세요
- 실습 기본 지역은 전주입니다. 도구 기본 인자를 그대로 쓰면 됩니다
"""

SAMPLE_QUESTIONS = [
    {
        "label": "A. 오늘·내일 날씨",
        "text": "오늘과 내일 전주 날씨를 알려줘. 강수 가능성이 있으면 방제도 함께 주의해야 하는지 짧게 말해줘.",
    },
    {
        "label": "B. 10일 전망",
        "text": "앞으로 10일간 전주 날씨 전망과 기온을 정리해줘.",
    },
    {
        "label": "C. 토마토 육묘·정식",
        "text": "토마토 육묘와 정식 시 주의사항을 농업기술길잡이 기준으로 요약해줘.",
    },
    {
        "label": "D. 토마토 10일 계획",
        "text": "앞으로 10일간 전주 날씨를 분석해서 토마토 농작업 계획을 세워줘.",
    },
    {
        "label": "E. 딸기(자료 없음)",
        "text": "딸기 재배에서 정식 후 물 관리 방법을 알려줘.",
    },
]


def load_keys() -> dict[str, str]:
    load_dotenv(ENV_PATH)
    keys = {
        "KMA_SHORT_TERM_KEY": os.getenv("KMA_SHORT_TERM_KEY") or "",
        "KMA_MID_TERM_KEY": os.getenv("KMA_MID_TERM_KEY") or "",
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY") or "",
    }
    missing = [name for name, value in keys.items() if not value]
    if missing:
        raise ValueError(f"다음 키를 {ENV_PATH}에서 찾을 수 없습니다: {', '.join(missing)}")
    return keys


def call_kma_api(url: str, service_key: str, extra_params: dict) -> dict:
    params = {
        "serviceKey": unquote(service_key),
        "pageNo": "1",
        "numOfRows": "1000",
        "dataType": "JSON",
        **extra_params,
    }
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    try:
        payload = response.json()
    except ValueError as exc:
        preview = response.text[:200].replace(unquote(service_key), "***")
        raise ValueError(f"JSON이 아닌 응답입니다. 미리보기: {preview}") from exc

    header = payload.get("response", {}).get("header", {})
    result_code = header.get("resultCode")
    if result_code != "00":
        raise RuntimeError(f"기상청 API 오류: [{result_code}] {header.get('resultMsg')}")
    return payload["response"]


def extract_items(response_body: dict) -> list[dict]:
    items = response_body.get("body", {}).get("items", {}).get("item", [])
    if isinstance(items, dict):
        return [items]
    return items or []


def latest_vilage_base(now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now(KST)
    hours = [2, 5, 8, 11, 14, 17, 20, 23]
    candidates = []
    for hour in hours:
        announced = now.replace(hour=hour, minute=10, second=0, microsecond=0)
        candidates.append((announced, f"{hour:02d}00"))
    available = [(t, bt) for t, bt in candidates if now >= t]
    if available:
        base_dt, base_time = available[-1]
        return base_dt.strftime("%Y%m%d"), base_time
    yesterday = now - timedelta(days=1)
    return yesterday.strftime("%Y%m%d"), "2300"


def latest_ncst_base(now: datetime | None = None) -> tuple[str, str]:
    now = now or datetime.now(KST)
    base = now.replace(minute=0, second=0, microsecond=0)
    if now.minute < 10:
        base -= timedelta(hours=1)
    return base.strftime("%Y%m%d"), base.strftime("%H00")


def latest_mid_tmfc(now: datetime | None = None) -> str:
    now = now or datetime.now(KST)
    today_06 = now.replace(hour=6, minute=10, second=0, microsecond=0)
    today_18 = now.replace(hour=18, minute=10, second=0, microsecond=0)
    if now >= today_18:
        return now.strftime("%Y%m%d") + "1800"
    if now >= today_06:
        return now.strftime("%Y%m%d") + "0600"
    yesterday = now - timedelta(days=1)
    return yesterday.strftime("%Y%m%d") + "1800"


def _to_float(value) -> float | None:
    try:
        return float(str(value).replace("mm", "").strip())
    except (TypeError, ValueError):
        return None


def crop_from_filename(name: str) -> str | None:
    for keyword, crop in CROP_KEYWORDS:
        if keyword in name:
            return crop
    return None


def list_pdf_files() -> list[Path]:
    pdfs = [
        path
        for path in DATA_DIR.iterdir()
        if path.suffix.lower() == ".pdf" and crop_from_filename(path.name)
    ]
    return sorted(pdfs, key=lambda p: crop_from_filename(p.name) or p.name)


def load_pdf_documents(pdf_paths: list[Path]) -> list[Document]:
    docs: list[Document] = []
    for path in pdf_paths:
        crop = crop_from_filename(path.name)
        reader = PdfReader(str(path))
        for idx, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{3,}", "\n\n", text)
            if len(text) < 80:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={"crop": crop, "source": path.name, "page": idx},
                )
            )
    return docs


def _add_batch(store, batch, embeddings, start_idx: int):
    for attempt in range(8):
        try:
            if store is None:
                return FAISS.from_documents(batch, embeddings)
            store.add_documents(batch)
            return store
        except Exception as exc:
            msg = str(exc).lower()
            if "rate" in msg or "429" in msg:
                wait = 20 + attempt * 10
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"임베딩 rate limit 재시도 실패 (청크 {start_idx})")


def load_or_build_vectorstore(embeddings: OpenAIEmbeddings):
    index_file = VS_DIR / "index.faiss"
    if index_file.exists():
        return FAISS.load_local(
            str(VS_DIR),
            embeddings,
            allow_dangerous_deserialization=True,
        )

    pdf_files = list_pdf_files()
    if len(pdf_files) != 6:
        raise FileNotFoundError("농업기술길잡이 PDF 6종을 같은 폴더에서 찾지 못했습니다.")
    raw_docs = load_pdf_documents(pdf_files)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(raw_docs)
    store = None
    batch_size = 40
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        store = _add_batch(store, batch, embeddings, start + 1)
        time.sleep(2)
    VS_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(VS_DIR))
    return store


def format_crop_hits(docs: list[Document]) -> str:
    if not docs:
        return "검색된 농업기술길잡이 문맥이 없습니다. 없는 내용은 만들어 내지 마세요."
    blocks = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        excerpt = doc.page_content.strip().replace("\n", " ")
        if len(excerpt) > 700:
            excerpt = excerpt[:700] + "..."
        blocks.append(
            f"[{i}] 작물={meta.get('crop')} | 파일={meta.get('source')} | 페이지={meta.get('page')}\n{excerpt}"
        )
    return "아래 발췌만 근거로 답하세요. 발췌에 없는 내용은 추측하지 마세요.\n\n" + "\n\n".join(blocks)


def summarize_short_term(nx: int, ny: int, short_key: str) -> str:
    vilage_date, vilage_time = latest_vilage_base()
    ncst_date, ncst_time = latest_ncst_base()

    ncst_items = extract_items(
        call_kma_api(
            f"{SHORT_TERM_BASE}/getUltraSrtNcst",
            short_key,
            {"base_date": ncst_date, "base_time": ncst_time, "nx": nx, "ny": ny},
        )
    )
    now_obs = {}
    for item in ncst_items:
        cat = item.get("category")
        val = item.get("obsrValue")
        if cat == "T1H":
            now_obs["기온(℃)"] = val
        elif cat == "REH":
            now_obs["습도(%)"] = val
        elif cat == "WSD":
            now_obs["풍속(m/s)"] = val
        elif cat == "RN1":
            now_obs["1시간강수"] = val
        elif cat == "PTY":
            now_obs["강수형태"] = PTY_CODE.get(str(val), val)

    fcst_items = extract_items(
        call_kma_api(
            f"{SHORT_TERM_BASE}/getVilageFcst",
            short_key,
            {"base_date": vilage_date, "base_time": vilage_time, "nx": nx, "ny": ny},
        )
    )
    by_slot: dict[tuple[str, str], dict] = {}
    for item in fcst_items:
        key = (str(item.get("fcstDate")), str(item.get("fcstTime")))
        slot = by_slot.setdefault(key, {"date": key[0], "time": key[1]})
        cat = item.get("category")
        val = item.get("fcstValue")
        if cat == "TMP":
            slot["tmp"] = _to_float(val)
        elif cat == "POP":
            slot["pop"] = _to_float(val)
        elif cat == "REH":
            slot["reh"] = _to_float(val)
        elif cat == "SKY":
            slot["sky"] = SKY_CODE.get(str(val), val)
        elif cat == "PTY":
            slot["pty"] = PTY_CODE.get(str(val), val)
        elif cat == "PCP":
            slot["pcp"] = val
        elif cat == "WSD":
            slot["wsd"] = _to_float(val)

    daily: dict[str, dict] = {}
    hourly = []
    for (date, time_code), slot in sorted(by_slot.items()):
        d = daily.setdefault(
            date,
            {
                "date": f"{date[:4]}-{date[4:6]}-{date[6:]}",
                "tmps": [],
                "pops": [],
                "rehs": [],
                "skies": [],
                "ptys": [],
            },
        )
        if slot.get("tmp") is not None:
            d["tmps"].append(slot["tmp"])
        if slot.get("pop") is not None:
            d["pops"].append(slot["pop"])
        if slot.get("reh") is not None:
            d["rehs"].append(slot["reh"])
        if slot.get("sky"):
            d["skies"].append(slot["sky"])
        if slot.get("pty") and slot["pty"] != "없음":
            d["ptys"].append(slot["pty"])
        hh = int(time_code[:2])
        if hh % 3 == 0:
            hourly.append(
                {
                    "시각": f"{date[:4]}-{date[4:6]}-{date[6:]} {time_code[:2]}:00",
                    "기온(℃)": slot.get("tmp"),
                    "강수확률(%)": slot.get("pop"),
                    "하늘": slot.get("sky"),
                    "강수형태": slot.get("pty"),
                    "습도(%)": slot.get("reh"),
                }
            )

    daily_rows = []
    for date, d in sorted(daily.items()):
        daily_rows.append(
            {
                "날짜": d["date"],
                "최저기온(℃)": min(d["tmps"]) if d["tmps"] else None,
                "최고기온(℃)": max(d["tmps"]) if d["tmps"] else None,
                "최대강수확률(%)": max(d["pops"]) if d["pops"] else None,
                "평균습도(%)": round(sum(d["rehs"]) / len(d["rehs"]), 1) if d["rehs"] else None,
                "하늘상태": max(set(d["skies"]), key=d["skies"].count) if d["skies"] else None,
                "강수형태": ", ".join(sorted(set(d["ptys"]))) if d["ptys"] else "없음",
            }
        )

    payload = {
        "지역": LOCATION_NAME,
        "자료": "기상청 단기예보 조회서비스 (실황+단기예보)",
        "실황기준": f"{ncst_date} {ncst_time}",
        "예보발표": f"{vilage_date} {vilage_time}",
        "현재실황": now_obs,
        "일별요약": daily_rows,
        "시간별(3시간)": hourly[:24],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def summarize_mid_term(land_reg_id: str, ta_reg_id: str, mid_key: str) -> str:
    tm_fc = latest_mid_tmfc()
    announce_dt = datetime.strptime(tm_fc, "%Y%m%d%H%M").replace(tzinfo=KST)

    outlook = ""
    stn_id = LAND_TO_STN.get(land_reg_id, STN_ID)
    try:
        mid_items = extract_items(
            call_kma_api(
                f"{MID_TERM_BASE}/getMidFcst",
                mid_key,
                {"stnId": stn_id, "tmFc": tm_fc},
            )
        )
        if mid_items:
            outlook = mid_items[0].get("wfSv", "") or ""
    except Exception:
        outlook = "(중기전망 텍스트는 이 시각에 제공되지 않았습니다)"

    land_item = extract_items(
        call_kma_api(
            f"{MID_TERM_BASE}/getMidLandFcst",
            mid_key,
            {"regId": land_reg_id, "tmFc": tm_fc},
        )
    )[0]
    ta_item = extract_items(
        call_kma_api(
            f"{MID_TERM_BASE}/getMidTa",
            mid_key,
            {"regId": ta_reg_id, "tmFc": tm_fc},
        )
    )[0]

    days = []
    for day in range(3, 11):
        forecast_date = (announce_dt + timedelta(days=day)).strftime("%Y-%m-%d")
        if day <= 7:
            wf_am, wf_pm = land_item.get(f"wf{day}Am"), land_item.get(f"wf{day}Pm")
            pop_am, pop_pm = land_item.get(f"rnSt{day}Am"), land_item.get(f"rnSt{day}Pm")
        else:
            wf_am = wf_pm = land_item.get(f"wf{day}")
            pop_am = pop_pm = land_item.get(f"rnSt{day}")
        days.append(
            {
                "일차": f"+{day}일",
                "날짜": forecast_date,
                "날씨(오전)": wf_am,
                "날씨(오후)": wf_pm,
                "강수확률(오전)%": pop_am,
                "강수확률(오후)%": pop_pm,
                "최저기온(℃)": ta_item.get(f"taMin{day}"),
                "최고기온(℃)": ta_item.get(f"taMax{day}"),
            }
        )

    payload = {
        "지역": LOCATION_NAME,
        "자료": "기상청 중기예보 조회서비스 (육상+기온+전망)",
        "발표시각": tm_fc,
        "안내": "육상예보 구역과 기온예보 구역은 서로 다른 코드입니다. 1~3일 상세는 단기예보를 참고하세요.",
        "중기전망": outlook,
        "일별전망": days,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def message_text(msg) -> str:
    content = getattr(msg, "content", "")
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
            safe_args = {k: v for k, v in (args or {}).items() if "key" not in str(k).lower()}
            traces.append({"name": name, "label": TOOL_LABELS.get(name, name), "args": safe_args})
        role = type(msg).__name__
        if role == "ToolMessage":
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
    keys = load_keys()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=40)
    vectorstore = load_or_build_vectorstore(embeddings)

    @tool
    def get_short_term_weather(nx: int = NX, ny: int = NY) -> str:
        """전주 등 격자의 오늘~글피(약 1~3일) 상세 날씨를 조회합니다.
        기온, 강수확률, 하늘상태, 강수형태, 습도를 요약합니다.
        10일 농작업 계획의 D+0~2(오늘~모레) 구간은 반드시 이 도구로 채우세요.
        4일 이후 전망만 필요할 때는 중기예보 도구를 쓰세요.
        """
        try:
            return summarize_short_term(nx, ny, keys["KMA_SHORT_TERM_KEY"])
        except Exception as exc:
            return f"단기예보 조회 실패: {type(exc).__name__}: {exc}"

    @tool
    def get_mid_term_weather(land_reg_id: str = LAND_REG_ID, ta_reg_id: str = TA_REG_ID) -> str:
        """중기예보로 +4일~+10일 날씨·강수확률·최저기온·최고기온을 조회합니다.
        육상예보 구역코드와 기온예보 구역코드는 서로 다릅니다.
        오늘~모레 상세는 이 도구에 없습니다. 10일 계획에서는 단기도 함께 호출하세요.
        """
        try:
            return summarize_mid_term(land_reg_id, ta_reg_id, keys["KMA_MID_TERM_KEY"])
        except Exception as exc:
            return f"중기예보 조회 실패: {type(exc).__name__}: {exc}"

    @tool
    def search_crop_guide(query: str, crop: str = "") -> str:
        """농촌진흥청 농업기술길잡이 PDF에서 작물 재배 지식을 검색합니다.
        가능한 작물: 토마토, 고추, 수박, 참외, 고구마, 사과.
        목록에 없는 작물은 다른 작물로 대체하지 말고 자료 없음을 반환합니다.
        """
        crop = (crop or "").strip()
        if crop and crop not in ALLOWED_CROPS:
            return (
                f"제공된 농업기술길잡이 PDF에 '{crop}' 자료가 없습니다. "
                f"가능한 작물: {', '.join(ALLOWED_CROPS)}. "
                "다른 작물 내용을 해당 작물인 것처럼 사용하지 마세요. "
                "주어진 자료만으로는 알 수 없다고 답하세요."
            )
        try:
            search_kwargs = {"k": 5}
            if crop:
                search_kwargs["filter"] = {"crop": crop}
                search_kwargs["fetch_k"] = 40
            docs = vectorstore.similarity_search(query, **search_kwargs)
            if crop:
                docs = [d for d in docs if d.metadata.get("crop") == crop]
            return format_crop_hits(docs)
        except Exception as exc:
            return f"작물 길잡이 검색 실패: {type(exc).__name__}: {exc}"

    agent = create_agent(
        model=llm,
        tools=[get_short_term_weather, get_mid_term_weather, search_crop_guide],
        system_prompt=SYSTEM_PROMPT,
    )
    return {
        "agent": agent,
        "vector_count": int(vectorstore.index.ntotal),
        "location": LOCATION_NAME,
        "crops": ALLOWED_CROPS,
        "pdf_count": len(list_pdf_files()),
        "index_ready": (VS_DIR / "index.faiss").exists(),
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
