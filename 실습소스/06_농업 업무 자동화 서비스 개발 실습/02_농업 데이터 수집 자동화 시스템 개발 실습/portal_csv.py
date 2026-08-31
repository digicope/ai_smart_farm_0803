# -*- coding: utf-8 -*-
"""공공데이터포털 파일데이터 다운로드 + 실습용 CSV 정리.

Open API(serviceKey)는 쓰지 않는다.
로그인 없이 fileDownload.do 로 CSV/ZIP을 받는다.
"""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

DATA_DIR = Path(__file__).resolve().parent / "data"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://www.data.go.kr/",
}

DATASETS = {
    "pest_weather": {
        "public_data_pk": "15136768",
        "title": "농촌진흥청_병해충발생예측 활용 기상정보",
        "page": "https://www.data.go.kr/data/15136768/fileData.do",
        "raw_name": "raw_pest_weather.csv",
        "large": True,
    },
    "field_farm": {
        "public_data_pk": "15126334",
        "title": "농촌진흥청_노지 현장 농가 데이터",
        "page": "https://www.data.go.kr/data/15126334/fileData.do",
        "raw_name": "raw_field_farm.zip",
        "large": False,
    },
    "pest_sites": {
        "public_data_pk": "15123424",
        "title": "농촌진흥청_병해충조사지점현황정보",
        "page": "https://www.data.go.kr/data/15123424/fileData.do",
        "raw_name": "raw_pest_sites.csv",
        "large": False,
    },
    "pest_catalog": {
        "public_data_pk": "15151253",
        "title": "농림수산식품교육문화정보원_병해충",
        "page": "https://www.data.go.kr/data/15151253/fileData.do",
        "raw_name": "raw_pest_catalog.csv",
        "large": False,
    },
}

AGENT_TO_PORTAL = {
    "weather": ["pest_weather"],
    "farm": ["field_farm"],
    "growth": ["field_farm"],
    "pest_sites": ["pest_sites"],
    "pest_info": ["pest_catalog"],
    "all": ["pest_weather", "field_farm", "pest_sites", "pest_catalog"],
}

PREPARED = {
    "weather": DATA_DIR / "rda_weather_wanju_2023.csv",
    "farm": DATA_DIR / "rda_farm_info_2024.csv",
    "growth": DATA_DIR / "rda_growth_2024.csv",
    "pest_sites": DATA_DIR / "rda_pest_sites.csv",
    "pest_info": DATA_DIR / "rda_pest_catalog.csv",
}


def find_uddi(html: str) -> str:
    m = re.search(r"uddi:([0-9a-fA-F\-]{8,})", html)
    return m.group(1) if m else ""


def get_atch_file_id(public_data_pk: str, uddi: str = "") -> str:
    params = {
        "publicDataPk": public_data_pk,
        "atchFileId": "",
        "fileDetailSn": "1",
        "url": "/tcs/dss/selectFileDataDownload.do",
    }
    if uddi:
        params["publicDataDetailPk"] = f"uddi:{uddi}"
    r = requests.get(
        "https://www.data.go.kr/tcs/dss/selectFileDataDownload.do",
        params=params,
        headers=HEADERS,
        timeout=60,
    )
    r.raise_for_status()
    meta = r.json()
    atch = meta.get("atchFileId") or ""
    if not atch:
        raise RuntimeError(f"atchFileId를 찾지 못했습니다: {public_data_pk}")
    return str(atch)


def download_raw(portal_id: str, force: bool = False) -> tuple[Path, str]:
    item = DATASETS[portal_id]
    dest = DATA_DIR / item["raw_name"]
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if item.get("large") and PREPARED["weather"].exists() and not force:
        return dest, (
            f"기상 원본 약 85MB 재다운로드 생략. 정리본 {PREPARED['weather'].name} 사용. "
            "다시 받으려면 force=True"
        )

    page = requests.get(item["page"], headers=HEADERS, timeout=40)
    page.raise_for_status()
    uddi = find_uddi(page.text)
    atch = get_atch_file_id(item["public_data_pk"], uddi)
    r = requests.get(
        "https://www.data.go.kr/cmm/cmm/fileDownload.do",
        params={"atchFileId": atch, "fileDetailSn": "1", "insertDataPrcus": "N"},
        headers=HEADERS,
        timeout=180,
    )
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest, f"포털 다운로드 {dest.name} ({len(r.content)}바이트) atchFileId={atch}"


def read_csv_kr(path: Path, **kwargs) -> pd.DataFrame:
    last = None
    for enc in ("cp949", "euc-kr", "utf-8-sig", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc, **kwargs)
        except Exception as exc:
            last = exc
    raise last


def save_prepared(df: pd.DataFrame, name: str) -> Path:
    out = DATA_DIR / name
    df.to_csv(out, index=False, encoding="utf-8-sig")
    return out


def prepare_weather(raw: Path) -> str:
    df = read_csv_kr(raw)
    df = df[df["지점명"].astype(str).str.startswith("완주군")].copy()
    df["시간"] = pd.to_datetime(df["시간"], errors="coerce")
    df["obs_date"] = df["시간"].dt.strftime("%Y-%m-%d")
    for col in ("기온누적값", "상대습도", "강수량", "풍속", "엽면습윤율"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    daily = (
        df.groupby(["지점명", "obs_date"], as_index=False)
        .agg(
            tmin_c=("기온누적값", "min"),
            tmax_c=("기온누적값", "max"),
            tavg_c=("기온누적값", "mean"),
            humidity_pct=("상대습도", "mean"),
            rainfall_mm=("강수량", "sum"),
            wind_ms=("풍속", "mean"),
            leaf_wetness=("엽면습윤율", "mean"),
        )
        .round(2)
    )
    daily.insert(0, "sido", "전북")
    daily.insert(1, "sigungu", "완주군")
    daily = daily.rename(columns={"지점명": "station"})
    out = save_prepared(daily, "rda_weather_wanju_2023.csv")
    return f"정리 {out.name} {len(daily)}건"


def _zip_root() -> Path:
    hits = list((DATA_DIR / "rda_field_farm").rglob("공개용_2024_농가정보.csv"))
    if not hits:
        raise FileNotFoundError("ZIP에서 2024 농가정보 CSV를 찾지 못했습니다.")
    return hits[0].parents[2]


def extract_farm_zip(raw: Path) -> Path:
    extract = DATA_DIR / "rda_field_farm"
    extract.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(raw) as zf:
        zf.extractall(extract)
    return _zip_root()


def prepare_farm(root: Path) -> str:
    src = root / "2024" / "농가정보" / "공개용_2024_농가정보.csv"
    df = read_csv_kr(src, dtype=str)
    df = df.rename(
        columns={
            "연도": "year",
            "지역(도)": "sido",
            "시군": "sigungu",
            "농가명": "farm_id",
            "작목": "crop",
            "품종": "variety",
            "포장면적": "area_m2",
            "주간거리": "in_row_m",
            "조간거리": "between_row_m",
            "파종일자": "sow_date",
            "정식일자": "plant_date",
            "수확일자": "harvest_date",
            "총수확량": "yield_total",
            "비고": "note",
        }
    )
    out = save_prepared(df, "rda_farm_info_2024.csv")
    return f"정리 {out.name} {len(df)}건"


def prepare_growth(root: Path) -> str:
    base = root / "2024" / "생육기본"
    frames = []
    pepper = read_csv_kr(base / "공개용_생육기본_고추_24년.csv")
    pepper = pepper.rename(
        columns={
            "시도": "sido",
            "시군구": "sigungu",
            "품목": "crop",
            "농가명": "farm_id",
            "조사일": "survey_date",
            "개체번호": "plant_no",
            "초장": "plant_height_cm",
            "착과수": "fruit_count",
            "수확과수": "harvest_count",
            "비고": "note",
        }
    )
    frames.append(pepper)
    for fname, height_col, extra in [
        ("공개용_생육기본_밀_24년.csv", "초장", "수수"),
        ("공개용_생육기본_배추_24년.csv", "초장(엽장)", "엽수"),
        ("공개용_생육기본_양파_24년.csv", None, None),
        ("공개용_생육기본_콩_24년.csv", None, None),
    ]:
        path = base / fname
        if not path.exists():
            continue
        raw = read_csv_kr(path)
        if "시도" not in raw.columns:
            continue
        jb = raw[raw["시도"].astype(str).str.contains("전북", na=False)].copy()
        if jb.empty:
            continue
        height_src = height_col if height_col and height_col in jb.columns else next(
            (c for c in jb.columns if "초장" in c), None
        )
        out = pd.DataFrame(
            {
                "sido": jb["시도"],
                "sigungu": jb["시군구"],
                "crop": jb["품목"],
                "farm_id": jb["농가명"],
                "survey_date": jb["조사일"],
                "plant_no": jb["개체번호"] if "개체번호" in jb.columns else jb.get("조사구역"),
                "plant_height_cm": pd.to_numeric(jb[height_src], errors="coerce") if height_src else pd.NA,
                "fruit_count": pd.NA,
                "harvest_count": pd.NA,
                "note": jb["비고"] if "비고" in jb.columns else pd.NA,
            }
        )
        if extra and extra in jb.columns:
            out["note"] = out["note"].fillna("").astype(str)
            out["note"] = (out["note"] + " " + extra + "=" + jb[extra].astype(str)).str.strip()
        frames.append(out)
    growth_df = pd.concat(frames, ignore_index=True)
    path = save_prepared(growth_df, "rda_growth_2024.csv")
    return f"정리 {path.name} {len(growth_df)}건"


def prepare_pest_sites(raw: Path) -> str:
    df = read_csv_kr(raw)
    df = df.rename(
        columns={
            "조사년도": "year",
            "예찰구분": "survey_type",
            "작목": "crop",
            "조사구분": "survey_kind",
            "시도": "sido",
            "시군구": "sigungu",
            "읍면동": "eupmyeon",
            "경도좌표": "lon",
            "위도좌표": "lat",
            "지대구분": "zone",
            "면적(ha)": "area_ha",
        }
    )
    out = save_prepared(df, "rda_pest_sites.csv")
    return f"정리 {out.name} {len(df)}건"


def prepare_pest_catalog(raw: Path) -> str:
    df = read_csv_kr(raw)
    df = df.rename(
        columns={
            "국가농작물병해충관리시스템 고유번호": "ncpms_id",
            "병해충 검역 식별코드": "quarantine_code",
            "병해충상세정보": "detail",
            "상세정보 주소": "url",
            "식물검역정보시스템 병해충명": "pest_name",
        }
    )
    df["detail"] = df["detail"].astype(str).str.slice(0, 400)
    df = df.drop_duplicates(subset=["pest_name", "ncpms_id"])
    out = save_prepared(df, "rda_pest_catalog.csv")
    return f"정리 {out.name} {len(df)}건"


def download_and_prepare(source: str = "all", force: bool = False) -> str:
    source = (source or "all").strip().lower()
    if source not in AGENT_TO_PORTAL:
        return f"알 수 없는 source='{source}'. 가능: {', '.join(AGENT_TO_PORTAL)}"

    lines = ["공공데이터포털 파일 다운로드 (Open API 키 없음)"]
    done_portal: set[str] = set()
    farm_root: Path | None = None

    for portal_id in AGENT_TO_PORTAL[source]:
        if portal_id in done_portal:
            continue
        done_portal.add(portal_id)
        item = DATASETS[portal_id]
        lines.append(f"- {item['title']} {item['page']}")
        raw, msg = download_raw(portal_id, force=force)
        lines.append(f"  {msg}")

        if portal_id == "pest_weather":
            if PREPARED["weather"].exists() and not force and "생략" in msg:
                lines.append(f"  정리본 유지 {PREPARED['weather'].name}")
            else:
                if not raw.exists() or raw.suffix.lower() != ".csv":
                    lines.append("  기상 원본 CSV가 없어 정리 생략")
                else:
                    lines.append(f"  {prepare_weather(raw)}")
        elif portal_id == "field_farm":
            farm_root = extract_farm_zip(raw)
            lines.append(f"  {prepare_farm(farm_root)}")
            lines.append(f"  {prepare_growth(farm_root)}")
        elif portal_id == "pest_sites":
            lines.append(f"  {prepare_pest_sites(raw)}")
        elif portal_id == "pest_catalog":
            lines.append(f"  {prepare_pest_catalog(raw)}")

    return "\n".join(lines)
