# -*- coding: utf-8 -*-
"""농업기술 PDF RAG + 노지/스마트팜 CSV + DuckDuckGo 웹 검색 Agent.

OpenAI 키만 C:\\env\\.env 의 OPENAI_API_KEY 를 사용한다.
기상청·농업기술 데이터 플랫폼·유료 검색 API는 쓰지 않는다.
"""

from __future__ import annotations

import json
import os
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdf"
DATA_DIR = BASE_DIR / "data"
FIELD_ZIP = BASE_DIR / "노지_2024.zip"
SMART_ZIP = BASE_DIR / "스마트팜_2024.zip"
VS_DIR = Path(r"C:\env\agri_search_faiss")
ENV_PATH = r"C:\env\.env"

PDF_CROPS = ("토마토", "고추", "수박", "참외", "고구마", "딸기", "사과")
CROP_KEYWORDS = [
    ("토마토", "토마토"),
    ("고추", "고추"),
    ("수박", "수박"),
    ("참외", "참외"),
    ("고구마", "고구마"),
    ("딸기", "딸기"),
    ("사과", "사과"),
]
FIELD_CROPS = ("고추", "마늘", "밀", "배추", "사과", "양파", "옥수수", "콩", "포도")
SMART_CROPS = (
    "가지",
    "국화",
    "딸기",
    "방울토마토",
    "수박",
    "오이",
    "완숙토마토",
    "참외",
    "파프리카",
)
SMART_ENV_A = ("가지", "국화", "수박", "오이", "참외")
SMART_ENV_B = ("딸기", "방울토마토", "완숙토마토", "파프리카")
SKIP_NUM_COLS = {
    "연도",
    "연번",
    "작기",
    "농가명",
    "개체번호",
    "조사구역",
    "조사번호",
    "줄기번호",
    "화방번호",
    "과일번호",
    "결과지번호",
    "주지번호",
    "마디번호",
}

TOOL_LABELS = {
    "search_crop_guide": "농업기술 RAG",
    "query_field_csv": "노지 CSV",
    "query_smartfarm_csv": "스마트팜 CSV",
    "search_web": "웹 검색",
}

SAMPLE_QUESTIONS = [
    {
        "id": "A",
        "label": "A. PDF만",
        "text": "고구마 정식과 물 관리 요령을 농업기술길잡이 기준으로 요약해줘.",
        "expected": ["search_crop_guide"],
    },
    {
        "id": "B",
        "label": "B. 노지 CSV만",
        "text": (
            "2024년 노지 고추 생육을 요약해줘. 초장이나 착과가 어느 정도인지 "
            "지역별로 알려줘."
        ),
        "expected": ["query_field_csv"],
    },
    {
        "id": "C",
        "label": "C. 스마트팜 CSV만",
        "text": "스마트팜 완숙토마토 내부 온습도가 어떤지, 출하량도 같이 요약해줘.",
        "expected": ["query_smartfarm_csv"],
    },
    {
        "id": "D",
        "label": "D. RAG + CSV + 웹",
        "text": (
            "딸기 재배 주의사항을 길잡이에서 찾고, "
            "스마트팜 딸기 생육·환경 숫자와 최근 딸기 관련 이슈를 함께 정리해줘."
        ),
        "expected": ["search_crop_guide", "query_smartfarm_csv", "search_web"],
    },
    {
        "id": "E",
        "label": "E. 자료 없음",
        "text": (
            "벼 논물 수심이 적절한지, 2024년 노지 딸기 생육 초장과 "
            "스마트팜 사과 내부온도까지 알려줘."
        ),
        "expected": ["자료 없음 명시"],
    },
]

SYSTEM_PROMPT = """당신은 농업 정보 검색 Agent입니다.
재배 지식은 농업기술길잡이 PDF, 실측 숫자는 로컬 CSV, 최신 이슈는 웹 검색 Tool 결과만 근거로 답합니다.
없는 숫자·약제명·재배법·뉴스를 추측하지 마세요. 근거가 없으면 '주어진 자료만으로는 알 수 없다'고 말하세요.

[도구 선택 — 반드시 지키세요]
- 정식/육묘/시비/관수/병해충 방제 등 재배 매뉴얼 → search_crop_guide
- 2024 노지 농가·생육 숫자 → query_field_csv
- 스마트팜 온실 환경·생육·출하량·재배정보 → query_smartfarm_csv
- 올해/최근/시세/발생 소식 등 최신 정보 → search_web
- 질문과 무관한 도구는 호출하지 마세요.
- 고구마처럼 재배 요령만 물으면 search_crop_guide만 쓰세요. CSV·웹을 치지 마세요.
- 노지 고추 생육만 물으면 query_field_csv만 쓰세요. 스마트팜·웹을 치지 마세요.
- 스마트팜 온습도·출하량만 물으면 query_smartfarm_csv만 쓰세요.
- 길잡이 + 스마트팜 숫자 + 최근 이슈를 함께 물으면 세 도구를 모두 호출하세요.

[자료 범위]
- PDF 작물: 토마토, 고추, 수박, 참외, 고구마, 딸기, 사과
- 노지 CSV 작물: 고추, 마늘, 밀, 배추, 사과, 양파, 옥수수, 콩, 포도
- 스마트팜 CSV 작물: 가지, 국화, 딸기, 방울토마토, 수박, 오이, 완숙토마토, 참외, 파프리카
- PDF의 '토마토'와 CSV의 '완숙토마토'/'방울토마토'는 구분해서 적으세요.
- 노지에 딸기·토마토가 없습니다. 스마트팜에 사과·고구마가 없습니다.
- 벼는 PDF·CSV 어디에도 없습니다. 다른 작물로 대체하지 마세요.

[답변 형식]
- 숫자는 CSV Tool 결과만 사용하세요. 웹 스니펫 숫자로 대체하지 마세요.
- 재배법은 PDF 발췌만 사용하세요. PDF에 없으면 웹으로 매뉴얼을 대체하지 마세요.
- PDF를 썼으면 파일명과 페이지를 밝히세요.
- CSV를 썼으면 작물·건수·기간을 밝히세요.
- 웹을 썼으면 제목과 URL을 밝히세요.
- API 키를 출력하지 마세요.
"""

_CSV_CACHE: dict[str, pd.DataFrame] = {}


def load_keys() -> dict[str, str]:
    load_dotenv(ENV_PATH)
    key = os.getenv("OPENAI_API_KEY") or ""
    if not key:
        raise ValueError(f"OPENAI_API_KEY 를 {ENV_PATH}에서 찾을 수 없습니다.")
    return {"OPENAI_API_KEY": key}


def decode_zip_name(name: str) -> str:
    for enc in ("cp437", "latin1"):
        for dec in ("cp949", "euc-kr", "utf-8"):
            try:
                return name.encode(enc).decode(dec)
            except Exception:
                pass
    return name


def extract_zip(zip_path: Path, dest: Path) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            name = decode_zip_name(info.filename)
            target = dest / name
            if info.is_dir() or name.endswith("/"):
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                out.write(src.read())
            extracted.append(str(target.relative_to(dest)))
    return extracted


def ensure_extracted() -> dict[str, Path]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    field_root = DATA_DIR / "노지_2024"
    smart_root = DATA_DIR / "스마트팜_2024"
    if not FIELD_ZIP.exists():
        raise FileNotFoundError(f"압축 파일이 없습니다: {FIELD_ZIP}")
    if not SMART_ZIP.exists():
        raise FileNotFoundError(f"압축 파일이 없습니다: {SMART_ZIP}")
    if not (field_root / "농가정보_2024").exists():
        extract_zip(FIELD_ZIP, DATA_DIR)
    if not any(smart_root.rglob("생산_2024.csv")):
        extract_zip(SMART_ZIP, smart_root)
    return {"field": field_root, "smart": smart_root}


def field_root() -> Path:
    return ensure_extracted()["field"]


def smart_root() -> Path:
    return ensure_extracted()["smart"]


def read_csv_smart(path: Path, **kwargs) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("cp949", "utf-8-sig", "utf-8", "euc-kr"):
        try:
            return pd.read_csv(path, encoding=enc, dtype=str, **kwargs)
        except UnicodeDecodeError as exc:
            last_err = exc
            continue
    raise UnicodeDecodeError("csv", b"", 0, 1, f"{path} 인코딩을 읽지 못했습니다: {last_err}")


def cached_csv(path: Path) -> pd.DataFrame:
    key = str(path.resolve())
    if key not in _CSV_CACHE:
        _CSV_CACHE[key] = read_csv_smart(path)
    return _CSV_CACHE[key]


def to_num(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def crop_from_filename(name: str) -> str | None:
    for keyword, crop in CROP_KEYWORDS:
        if keyword in name:
            return crop
    return None


def list_pdf_files() -> list[Path]:
    if not PDF_DIR.exists():
        return []
    pdfs = [
        path
        for path in PDF_DIR.iterdir()
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
                time.sleep(20 + attempt * 10)
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
    if len(pdf_files) != 7:
        names = [p.name for p in pdf_files]
        raise FileNotFoundError(f"농업기술길잡이 PDF 7종을 pdf/에서 찾지 못했습니다: {names}")
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
        time.sleep(1.5)
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
    return "아래 발췌만 근거로 답하세요. 발췌에 없는 내용은 추측하지 마세요.\n\n" + "\n\n".join(
        blocks
    )


def find_col(df: pd.DataFrame, *names: str) -> str | None:
    cols = list(df.columns)
    for name in names:
        if name in cols:
            return name
        for col in cols:
            if col.replace(" ", "") == name.replace(" ", ""):
                return col
    return None


def region_summary(df: pd.DataFrame, limit: int = 8) -> str:
    sido = find_col(df, "시도", "도", "도명", "지역(도)")
    sgg = find_col(df, "시군구", "시군")
    parts = []
    if sido:
        top = df[sido].fillna("").astype(str).str.strip().value_counts().head(limit)
        parts.append("시도=" + ", ".join(f"{k}:{int(v)}" for k, v in top.items() if k))
    if sgg:
        top = df[sgg].fillna("").astype(str).str.strip().value_counts().head(limit)
        parts.append("시군=" + ", ".join(f"{k}:{int(v)}" for k, v in top.items() if k))
    return " / ".join(parts)


def date_range_of(df: pd.DataFrame) -> str:
    for name in ("조사일", "조사일자", "출하일자", "측정시간", "정식일자", "수확일자", "정식일"):
        col = find_col(df, name)
        if col is None:
            continue
        vals = df[col].dropna().astype(str).str.strip()
        vals = vals[vals.str.match(r"^\d{4}-\d{2}-\d{2}", na=False)]
        if vals.empty:
            continue
        return f"{vals.min()} ~ {vals.max()}"
    return ""


def numeric_stats(df: pd.DataFrame, preferred: list[str] | None = None, max_cols: int = 8) -> dict:
    stats: dict[str, dict] = {}
    candidates = preferred or []
    if not candidates:
        candidates = [
            c
            for c in df.columns
            if c not in SKIP_NUM_COLS and not any(k in c for k in ("일자", "시간", "명", "번호"))
        ]
    count = 0
    for col in candidates:
        real = find_col(df, col) if col not in df.columns else col
        if real is None or real not in df.columns:
            continue
        nums = to_num(df[real]).dropna()
        if nums.empty:
            continue
        stats[real] = {
            "건수": int(nums.shape[0]),
            "평균": round(float(nums.mean()), 2),
            "최소": round(float(nums.min()), 2),
            "최대": round(float(nums.max()), 2),
        }
        count += 1
        if count >= max_cols:
            break
    return stats


def filter_crop_rows(df: pd.DataFrame, crops: list[str]) -> pd.DataFrame:
    crop_col = find_col(df, "작목", "품목")
    if crop_col is None:
        return df.iloc[0:0]
    series = df[crop_col].fillna("").astype(str).str.strip()
    return df[series.isin(crops)]


def filter_region(df: pd.DataFrame, region: str) -> pd.DataFrame:
    region = (region or "").strip()
    if not region:
        return df
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        if col in ("시도", "시군구", "도", "도명", "시군", "지역(도)"):
            mask = mask | df[col].fillna("").astype(str).str.contains(region, na=False)
    return df[mask]


def filter_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    start_date = (start_date or "").strip()
    end_date = (end_date or "").strip()
    if not start_date and not end_date:
        return df
    date_col = find_col(df, "측정시간", "조사일자", "출하일자", "조사일", "정식일")
    if date_col is None:
        return df
    series = df[date_col].fillna("").astype(str)
    mask = pd.Series(True, index=df.index)
    if start_date:
        mask &= series >= start_date
    if end_date:
        mask &= series <= f"{end_date} 23:59:59"
    return df[mask]


def expand_smart_crops(crop: str) -> list[str]:
    crop = (crop or "").strip()
    if crop in ("토마토", "방울·완숙토마토"):
        return ["완숙토마토", "방울토마토"]
    if crop in SMART_CROPS:
        return [crop]
    return [crop]


def field_farm_path() -> Path:
    matches = list(field_root().rglob("*농가정보*.csv"))
    if not matches:
        raise FileNotFoundError("노지 농가정보 CSV를 찾지 못했습니다.")
    return matches[0]


def field_growth_path(crop: str) -> Path | None:
    target = field_root() / "생육기본_2024" / f"생육기본_{crop}_2024.csv"
    if target.exists():
        return target
    matches = list(field_root().rglob(f"*{crop}*.csv"))
    growth = [p for p in matches if "생육" in p.name]
    return growth[0] if growth else None


def summarize_field(crop: str, kind: str, region: str = "") -> str:
    crop = (crop or "").strip()
    kind = (kind or "growth").strip().lower()
    if kind in ("농가", "농가정보"):
        kind = "farm"
    if kind in ("생육", "생육기본"):
        kind = "growth"
    if kind not in ("farm", "growth"):
        return "kind 는 farm 또는 growth 만 사용할 수 있습니다."
    if not crop:
        return "작물명(crop)이 필요합니다."

    if crop not in FIELD_CROPS:
        return (
            f"노지_2024 CSV에 '{crop}' 자료가 없습니다. "
            f"가능한 작물: {', '.join(FIELD_CROPS)}. "
            "다른 작물 숫자로 대체하지 마세요. 주어진 자료만으로는 알 수 없다고 답하세요."
        )

    if kind == "farm":
        df = cached_csv(field_farm_path())
        df = filter_crop_rows(df, [crop])
        df = filter_region(df, region)
        if df.empty:
            return f"노지 농가정보에서 '{crop}' (지역='{region}') 행이 없습니다."
        payload = {
            "자료": "노지_2024 농가정보",
            "작물": crop,
            "건수": int(len(df)),
            "기간": date_range_of(df),
            "지역": region_summary(df),
            "통계": numeric_stats(df, ["포장면적", "총수확량", "주간거리", "조간거리"]),
        }
        variety = find_col(df, "품종")
        if variety:
            top = df[variety].fillna("").astype(str).str.strip().value_counts().head(5)
            payload["주요품종"] = {k: int(v) for k, v in top.items() if k}
        return json.dumps(payload, ensure_ascii=False, indent=2)

    path = field_growth_path(crop)
    if path is None:
        return (
            f"노지 생육기본 CSV에 '{crop}' 파일이 없습니다. "
            "다른 작물 숫자로 대체하지 마세요."
        )
    df = cached_csv(path)
    df = filter_crop_rows(df, [crop]) if find_col(df, "품목", "작목") else df
    df = filter_region(df, region)
    if df.empty:
        return f"노지 생육에서 '{crop}' (지역='{region}') 행이 없습니다."
    preferred = [
        "초장",
        "착과수",
        "수확과수",
        "엽수",
        "구고",
        "구폭",
        "주중",
        "초장(엽장)",
        "수고",
        "수폭",
        "신초길이",
        "과중",
        "구직경",
        "생구무게",
        "간장",
        "분지수",
        "종실중",
        "당도",
    ]
    payload = {
        "자료": f"노지_2024 생육기본 ({path.name})",
        "작물": crop,
        "건수": int(len(df)),
        "기간": date_range_of(df),
        "지역": region_summary(df),
        "통계": numeric_stats(df, preferred),
    }
    sido = find_col(df, "시도", "도", "도명", "지역(도)")
    if sido:
        grouped = []
        num_col = find_col(df, "초장", "초장(엽장)", "수고", "간장")
        fruit_col = find_col(df, "착과수", "과중", "생구무게")
        for name, part in df.groupby(df[sido].fillna("").astype(str).str.strip()):
            if not name:
                continue
            row = {"시도": name, "건수": int(len(part))}
            if num_col:
                nums = to_num(part[num_col]).dropna()
                if not nums.empty:
                    row[f"{num_col}_평균"] = round(float(nums.mean()), 2)
            if fruit_col:
                nums = to_num(part[fruit_col]).dropna()
                if not nums.empty:
                    row[f"{fruit_col}_평균"] = round(float(nums.mean()), 2)
            grouped.append(row)
        payload["시도별"] = grouped[:10]
    return json.dumps(payload, ensure_ascii=False, indent=2)


def smart_file(pattern: str) -> Path | None:
    matches = list(smart_root().rglob(pattern))
    return matches[0] if matches else None


def env_files_for(crops: list[str]) -> list[Path]:
    files: list[Path] = []
    if any(c in SMART_ENV_A for c in crops):
        found = [p for p in smart_root().rglob("환경_2024*.csv") if "가지" in p.name or "오이" in p.name]
        files.extend(found)
    if any(c in SMART_ENV_B for c in crops):
        found = [p for p in smart_root().rglob("환경_2024*.csv") if "딸기" in p.name or "파프리카" in p.name]
        files.extend(found)
    # 파일명 매칭이 실패하면 환경 CSV 전부
    if not files:
        files = list(smart_root().rglob("환경_2024*.csv"))
    # 중복 제거
    uniq = []
    seen = set()
    for path in files:
        key = str(path.resolve())
        if key not in seen:
            uniq.append(path)
            seen.add(key)
    return uniq


def summarize_env_chunks(paths: list[Path], crops: list[str], region: str, start_date: str, end_date: str) -> pd.DataFrame:
    frames = []
    use_cols_hint = None
    for path in paths:
        for enc in ("cp949", "utf-8-sig", "utf-8"):
            try:
                reader = pd.read_csv(path, encoding=enc, dtype=str, chunksize=60000)
                break
            except UnicodeDecodeError:
                reader = None
        if reader is None:
            continue
        for chunk in reader:
            crop_col = find_col(chunk, "품목")
            if crop_col is None:
                continue
            part = chunk[chunk[crop_col].fillna("").astype(str).str.strip().isin(crops)]
            part = filter_region(part, region)
            part = filter_dates(part, start_date, end_date)
            if part.empty:
                continue
            keep = [
                c
                for c in part.columns
                if c
                in {
                    crop_col,
                    find_col(part, "도", "도명"),
                    find_col(part, "시군", "시군구"),
                    find_col(part, "측정시간"),
                    find_col(part, "온도_내부"),
                    find_col(part, "상대습도_내부"),
                    find_col(part, "잔존CO2", "잔존 CO2"),
                    find_col(part, "토양온도"),
                    find_col(part, "일사량_외부"),
                    find_col(part, "온도_외부"),
                }
                and c is not None
            ]
            frames.append(part[keep].copy())
            if sum(len(f) for f in frames) >= 120000:
                break
        if sum(len(f) for f in frames) >= 120000:
            break
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def summarize_smart(crop: str, kind: str, region: str = "", start_date: str = "", end_date: str = "") -> str:
    crop = (crop or "").strip()
    kind = (kind or "").strip().lower()
    kind_map = {
        "재배": "cultivation",
        "재배정보": "cultivation",
        "환경": "environment",
        "생육": "growth",
        "생산": "production",
        "출하": "production",
    }
    kind = kind_map.get(kind, kind)
    if kind not in ("cultivation", "environment", "growth", "production"):
        return "kind 는 cultivation / environment / growth / production 중 하나여야 합니다."
    if not crop:
        return "작물명(crop)이 필요합니다."

    crops = expand_smart_crops(crop)
    unknown = [c for c in crops if c not in SMART_CROPS]
    if unknown:
        return (
            f"스마트팜_2024 CSV에 '{crop}' 자료가 없습니다. "
            f"가능한 작물: {', '.join(SMART_CROPS)}. "
            "다른 작물 숫자로 대체하지 마세요. 주어진 자료만으로는 알 수 없다고 답하세요."
        )

    if kind == "cultivation":
        path = smart_file("재배정보_2024.xlsx")
        if path is None:
            return "스마트팜 재배정보 xlsx를 찾지 못했습니다."
        df = pd.read_excel(path, dtype=str)
        df = filter_crop_rows(df, crops)
        df = filter_region(df, region)
        if df.empty:
            return f"스마트팜 재배정보에서 '{', '.join(crops)}' 행이 없습니다."
        payload = {
            "자료": "스마트팜_2024 재배정보",
            "작물": crops,
            "건수": int(len(df)),
            "기간": date_range_of(df),
            "지역": region_summary(df),
            "통계": numeric_stats(df, ["전체면적", "식부면적", "재식밀도"]),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    if kind == "production":
        path = smart_file("생산_2024.csv")
        if path is None:
            return "스마트팜 생산 CSV를 찾지 못했습니다."
        df = cached_csv(path)
        df = filter_crop_rows(df, crops)
        df = filter_region(df, region)
        df = filter_dates(df, start_date, end_date)
        if df.empty:
            return f"스마트팜 생산에서 '{', '.join(crops)}' 행이 없습니다."
        payload = {
            "자료": "스마트팜_2024 생산",
            "작물": crops,
            "건수": int(len(df)),
            "기간": date_range_of(df),
            "지역": region_summary(df),
            "통계": numeric_stats(df, ["총출하량", "판매금액"]),
        }
        crop_col = find_col(df, "품목")
        if crop_col and len(crops) > 1:
            by_crop = []
            for name, part in df.groupby(df[crop_col].fillna("").astype(str).str.strip()):
                qty = to_num(part[find_col(part, "총출하량")]).dropna() if find_col(part, "총출하량") else pd.Series(dtype=float)
                by_crop.append(
                    {
                        "품목": name,
                        "건수": int(len(part)),
                        "총출하량_합": round(float(qty.sum()), 2) if not qty.empty else None,
                    }
                )
            payload["품목별"] = by_crop
        return json.dumps(payload, ensure_ascii=False, indent=2)

    if kind == "growth":
        blocks = []
        for name in crops:
            path = smart_file(f"생육_{name}_2024.csv")
            if path is None:
                blocks.append({"작물": name, "결과": "생육 CSV 없음"})
                continue
            df = cached_csv(path)
            df = filter_crop_rows(df, [name]) if find_col(df, "품목") else df
            df = filter_region(df, region)
            df = filter_dates(df, start_date, end_date)
            if df.empty:
                blocks.append({"작물": name, "결과": "해당 조건 행 없음"})
                continue
            preferred = [
                "초장",
                "엽장",
                "엽폭",
                "엽수",
                "줄기굵기",
                "꽃수",
                "화방별착과수",
                "화방별꽃수",
                "관부직경",
                "마디수",
                "착과율",
            ]
            blocks.append(
                {
                    "작물": name,
                    "파일": path.name,
                    "건수": int(len(df)),
                    "기간": date_range_of(df),
                    "지역": region_summary(df),
                    "통계": numeric_stats(df, preferred),
                }
            )
        if all(b.get("결과") for b in blocks):
            return json.dumps(
                {"자료": "스마트팜_2024 생육", "작물": crops, "결과": blocks},
                ensure_ascii=False,
                indent=2,
            )
        return json.dumps({"자료": "스마트팜_2024 생육", "작물": crops, "요약": blocks}, ensure_ascii=False, indent=2)

    # environment
    paths = env_files_for(crops)
    if not paths:
        return "스마트팜 환경 CSV를 찾지 못했습니다."
    df = summarize_env_chunks(paths, crops, region, start_date, end_date)
    if df.empty:
        return (
            f"스마트팜 환경에서 '{', '.join(crops)}' 행이 없습니다. "
            "다른 작물 온습도로 대체하지 마세요."
        )
    preferred = ["온도_내부", "상대습도_내부", "잔존CO2", "잔존 CO2", "토양온도", "일사량_외부", "온도_외부"]
    payload = {
        "자료": "스마트팜_2024 환경 (필터·요약, 원문 전체 아님)",
        "작물": crops,
        "건수": int(len(df)),
        "기간": date_range_of(df),
        "지역": region_summary(df),
        "통계": numeric_stats(df, preferred),
        "안내": "환경 CSV는 대용량이므로 품목 필터 후 평균·최소·최대만 반환했습니다.",
    }
    crop_col = find_col(df, "품목")
    if crop_col and len(crops) > 1:
        by_crop = []
        tcol = find_col(df, "온도_내부")
        hcol = find_col(df, "상대습도_내부")
        for name, part in df.groupby(df[crop_col].fillna("").astype(str).str.strip()):
            row = {"품목": name, "건수": int(len(part))}
            if tcol:
                nums = to_num(part[tcol]).dropna()
                if not nums.empty:
                    row["내부온도_평균"] = round(float(nums.mean()), 2)
            if hcol:
                nums = to_num(part[hcol]).dropna()
                if not nums.empty:
                    row["내부습도_평균"] = round(float(nums.mean()), 2)
            by_crop.append(row)
        payload["품목별"] = by_crop
    return json.dumps(payload, ensure_ascii=False, indent=2)


def run_web_search(query: str, max_results: int = 5) -> str:
    query = (query or "").strip()
    if not query:
        return "검색어가 비었습니다."
    items = []
    last_err = ""
    try:
        from ddgs import DDGS

        items = list(DDGS().text(query, max_results=max_results, region="kr-kr"))
    except Exception as exc:
        last_err = f"{type(exc).__name__}: {exc}"
        try:
            from duckduckgo_search import DDGS as LegacyDDGS

            items = list(LegacyDDGS().text(query, max_results=max_results, region="kr-kr"))
            last_err = ""
        except Exception as exc2:
            last_err = f"{last_err} / {type(exc2).__name__}: {exc2}"
    if not items:
        return f"검색 결과 없음. {last_err}".strip()
    lines = ["웹 검색은 최신 이슈 보강용입니다. 재배 매뉴얼·실측 숫자로 쓰지 마세요.", ""]
    for i, item in enumerate(items[:max_results], start=1):
        title = item.get("title") or item.get("source") or ""
        url = item.get("href") or item.get("url") or ""
        body = (item.get("body") or item.get("snippet") or "").replace("\n", " ")
        if len(body) > 220:
            body = body[:220] + "..."
        lines.append(f"[{i}] 제목: {title}\nURL: {url}\n발췌: {body}")
    return "\n\n".join(lines)


def inventory_tables() -> pd.DataFrame:
    ensure_extracted()
    rows = []
    farm = field_farm_path()
    df = read_csv_smart(farm, nrows=3)
    full = cached_csv(farm)
    crop_col = find_col(full, "작목", "품목")
    crops = sorted(full[crop_col].dropna().astype(str).str.strip().unique()) if crop_col else []
    rows.append(
        {
            "구분": "노지 농가",
            "파일": farm.name,
            "행 수": len(full),
            "주요 컬럼": ", ".join(list(full.columns)[:8]),
            "작물": ", ".join(crops),
            "기간 힌트": date_range_of(full),
        }
    )
    for path in sorted((field_root() / "생육기본_2024").glob("*.csv")):
        sample = read_csv_smart(path, nrows=2)
        n = sum(1 for _ in open(path, encoding="cp949", errors="ignore")) - 1
        crop = path.stem.replace("생육기본_", "").replace("_2024", "")
        rows.append(
            {
                "구분": "노지 생육",
                "파일": path.name,
                "행 수": max(n, 0),
                "주요 컬럼": ", ".join(list(sample.columns)[:8]),
                "작물": crop,
                "기간 힌트": date_range_of(sample) or "(전체는 Tool에서 집계)",
            }
        )
    xlsx = smart_file("재배정보_2024.xlsx")
    if xlsx:
        df = pd.read_excel(xlsx, dtype=str)
        crop_col = find_col(df, "품목")
        crops = sorted(df[crop_col].dropna().astype(str).str.strip().unique()) if crop_col else []
        rows.append(
            {
                "구분": "스마트팜 재배정보",
                "파일": xlsx.name,
                "행 수": len(df),
                "주요 컬럼": ", ".join(list(df.columns)[:8]),
                "작물": ", ".join(crops),
                "기간 힌트": date_range_of(df),
            }
        )
    prod = smart_file("생산_2024.csv")
    if prod:
        sample = read_csv_smart(prod, nrows=2)
        n = sum(1 for _ in open(prod, encoding="cp949", errors="ignore")) - 1
        rows.append(
            {
                "구분": "스마트팜 생산",
                "파일": prod.name,
                "행 수": max(n, 0),
                "주요 컬럼": ", ".join(list(sample.columns)),
                "작물": "참외, 오이, 완숙토마토, 방울토마토, 딸기 등",
                "기간 힌트": "2024~2025 출하",
            }
        )
    for path in sorted((smart_root()).rglob("생육_*_2024.csv")):
        sample = read_csv_smart(path, nrows=2)
        crop = path.stem.replace("생육_", "").replace("_2024", "")
        rows.append(
            {
                "구분": "스마트팜 생육",
                "파일": path.name,
                "행 수": "(Tool에서 집계)",
                "주요 컬럼": ", ".join(list(sample.columns)[:8]),
                "작물": crop,
                "기간 힌트": date_range_of(sample) or "2024~2025",
            }
        )
    for path in sorted(smart_root().rglob("환경_2024*.csv")):
        sample = read_csv_smart(path, nrows=2)
        rows.append(
            {
                "구분": "스마트팜 환경",
                "파일": path.name,
                "행 수": "대용량(필터 후 요약)",
                "주요 컬럼": ", ".join(list(sample.columns)[:8]),
                "작물": path.stem.replace("환경_2024", ""),
                "기간 힌트": date_range_of(sample) or "2024~2025",
            }
        )
    return pd.DataFrame(rows)


def sample_heads() -> dict[str, pd.DataFrame]:
    ensure_extracted()
    out = {}
    farm = cached_csv(field_farm_path())
    out["노지 농가 샘플"] = farm.head(2)
    pepper = field_growth_path("고추")
    if pepper:
        out["노지 고추 생육 샘플"] = cached_csv(pepper).head(2)
    env = env_files_for(["완숙토마토"])
    if env:
        out["스마트팜 환경 head"] = read_csv_smart(env[0], nrows=2)
    return out


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
    load_keys()
    ensure_extracted()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", chunk_size=40)
    vectorstore = load_or_build_vectorstore(embeddings)

    @tool
    def search_crop_guide(query: str, crop: str = "") -> str:
        """농촌진흥청 농업기술길잡이 PDF에서 작물 재배 지식을 검색합니다.
        가능한 작물: 토마토, 고추, 수박, 참외, 고구마, 딸기, 사과.
        목록에 없는 작물은 다른 작물로 대체하지 말고 자료 없음을 반환합니다.
        재배/정식/육묘/시비/관수/병해충 매뉴얼에 사용하세요.
        검색된 발췌에 없는 재배법·약제명·수치를 만들지 마세요.
        """
        crop = (crop or "").strip()
        if crop and crop not in PDF_CROPS:
            return (
                f"제공된 농업기술길잡이 PDF에 '{crop}' 자료가 없습니다. "
                f"가능한 작물: {', '.join(PDF_CROPS)}. "
                "다른 작물 내용을 해당 작물인 것처럼 사용하지 마세요. "
                "주어진 자료만으로는 알 수 없다고 답하세요."
            )
        try:
            q = f"{crop} {query}".strip() if crop else query
            docs = vectorstore.similarity_search(q, k=12)
            if crop:
                docs = [d for d in docs if d.metadata.get("crop") == crop][:5]
            else:
                docs = docs[:5]
            return format_crop_hits(docs)
        except Exception as exc:
            return f"작물 길잡이 검색 실패: {type(exc).__name__}: {exc}"

    @tool
    def query_field_csv(crop: str, kind: str = "growth", region: str = "") -> str:
        """2024 노지 현장 농가정보·생육기본 CSV를 조회해 요약 통계를 반환합니다.
        kind: farm(농가정보) 또는 growth(생육).
        작물: 고추, 마늘, 밀, 배추, 사과, 양파, 옥수수, 콩, 포도.
        노지에 없는 작물(딸기, 토마토, 벼 등)은 자료 없음을 반환합니다.
        원문 전체를 반환하지 않습니다.
        """
        try:
            return summarize_field(crop, kind, region)
        except Exception as exc:
            return f"노지 CSV 조회 실패: {type(exc).__name__}: {exc}"

    @tool
    def query_smartfarm_csv(
        crop: str,
        kind: str,
        region: str = "",
        start_date: str = "",
        end_date: str = "",
    ) -> str:
        """2024 스마트팜 재배정보·환경·생육·생산 CSV를 조회해 요약 통계를 반환합니다.
        kind: cultivation / environment / growth / production
        작물: 가지, 국화, 딸기, 방울토마토, 수박, 오이, 완숙토마토, 참외, 파프리카.
        crop='토마토'이면 완숙토마토와 방울토마토를 구분해 조회합니다.
        스마트팜에 없는 작물(사과, 고구마, 벼 등)은 자료 없음을 반환합니다.
        환경 원문 수십만 행은 반환하지 않고 평균·최소·최대·건수만 반환합니다.
        """
        try:
            return summarize_smart(crop, kind, region, start_date, end_date)
        except Exception as exc:
            return f"스마트팜 CSV 조회 실패: {type(exc).__name__}: {exc}"

    @tool
    def search_web(query: str) -> str:
        """API 키가 필요 없는 DuckDuckGo로 최신 농업 이슈를 검색합니다.
        재배 매뉴얼은 PDF, 실측 숫자는 CSV를 쓰고
        웹 검색은 올해 이슈·정책·시세·병해충 경보 등 최신 정보에만 쓰세요.
        검색 스니펫의 숫자를 CSV 실측처럼 쓰지 마세요.
        """
        try:
            return run_web_search(query)
        except Exception as exc:
            return f"검색 결과 없음. {type(exc).__name__}: {exc}"

    agent = create_agent(
        model=llm,
        tools=[search_crop_guide, query_field_csv, query_smartfarm_csv, search_web],
        system_prompt=SYSTEM_PROMPT,
    )
    return {
        "agent": agent,
        "tools": {
            "search_crop_guide": search_crop_guide,
            "query_field_csv": query_field_csv,
            "query_smartfarm_csv": query_smartfarm_csv,
            "search_web": search_web,
        },
        "vector_count": int(vectorstore.index.ntotal),
        "pdf_count": len(list_pdf_files()),
        "index_ready": (VS_DIR / "index.faiss").exists(),
        "crops_pdf": PDF_CROPS,
        "crops_field": FIELD_CROPS,
        "crops_smart": SMART_CROPS,
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
