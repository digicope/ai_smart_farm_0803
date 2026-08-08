"""
AI 영농 상담 챗봇 답안 예제 (Function Calling 중심)

- OpenAI Chat Completions API의 tools / Function Calling 사용
- AI Agent 프레임워크 미사용
- 지식 조회는 farm_knowledge.json 기반

실행:
    streamlit run AI_영농_상담_챗봇_답안예제.py

또는 제출용으로 이름을 바꿔 실행:
    streamlit run app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

# ---------------------------------------------------------------------------
# 환경 설정
# ---------------------------------------------------------------------------
# 로컬 .env 또는 수업용 공통 경로를 순차로 시도
load_dotenv()
load_dotenv("C:/env/.env")

API_KEY = os.getenv("OPENAI_API_KEY")
MODEL = "gpt-4o-mini"
DATA_PATH = Path(__file__).resolve().parent / "farm_knowledge.json"

SYSTEM_PROMPT = (
    "당신은 스마트 농업 현장의 영농 상담 전문가입니다. "
    "함수로 조회한 사실 정보를 바탕으로, 농민이 바로 실행할 수 있는 조언을 "
    "간단하고 친절하게 설명하세요. "
    "함수 결과에 없는 내용은 단정하지 말고, 추가 확인이 필요하다고 안내하세요. "
    "약제명과 시비량은 등록 기준과 현장 상태에 따라 달라질 수 있음을 함께 알리세요."
)


# ---------------------------------------------------------------------------
# 지식 데이터 / 실제 함수
# ---------------------------------------------------------------------------
@st.cache_data
def load_knowledge() -> dict:
    with open(DATA_PATH, encoding="utf-8") as f:
        return json.load(f)


def get_weather(region: str) -> dict:
    """지역 날씨 및 영농 유의 정보를 조회한다."""
    knowledge = load_knowledge()
    weather = knowledge["weather"].get(region)

    if not weather:
        available = ", ".join(knowledge["weather"].keys())
        return {
            "found": False,
            "region": region,
            "message": f"'{region}' 날씨 데이터가 없습니다. 가능 지역: {available}",
        }

    return {
        "found": True,
        "region": region,
        "temperature_c": weather["temperature_c"],
        "humidity_pct": weather["humidity_pct"],
        "rain_prob_pct": weather["rain_prob_pct"],
        "wind": weather["wind"],
        "farming_note": weather["farming_note"],
    }


def get_disease_info(crop: str, symptom: str) -> dict:
    """작물과 증상으로 병해충 정보를 조회한다."""
    knowledge = load_knowledge()
    symptom_norm = symptom.replace(" ", "")
    candidates = []

    for item in knowledge["diseases"]:
        if item["crop"] != crop:
            continue
        for keyword in item["symptom_keywords"]:
            key = keyword.replace(" ", "")
            if key in symptom_norm or symptom_norm in key:
                candidates.append(item)
                break

    if not candidates:
        return {
            "found": False,
            "crop": crop,
            "symptom": symptom,
            "message": "일치하는 병해충 정보가 없습니다. 증상과 작물명을 조금 더 구체적으로 알려주세요.",
        }

    best = candidates[0]
    return {
        "found": True,
        "crop": crop,
        "symptom": symptom,
        "disease_name": best["disease_name"],
        "cause": best["cause"],
        "treatment": best["treatment"],
    }


def get_fertilizer_recommend(crop: str, growth_stage: str) -> dict:
    """작물과 생육 단계에 맞는 비료를 추천한다."""
    knowledge = load_knowledge()

    for item in knowledge["fertilizers"]:
        if item["crop"] == crop and item["growth_stage"] == growth_stage:
            return {
                "found": True,
                "crop": crop,
                "growth_stage": growth_stage,
                "recommend": item["recommend"],
                "amount_guide": item["amount_guide"],
                "caution": item["caution"],
            }

    stages = sorted(
        {x["growth_stage"] for x in knowledge["fertilizers"] if x["crop"] == crop}
    )
    return {
        "found": False,
        "crop": crop,
        "growth_stage": growth_stage,
        "message": (
            f"'{crop}'의 '{growth_stage}' 추천 정보가 없습니다."
            + (f" 가능한 단계: {', '.join(stages)}" if stages else " 해당 작물 데이터가 없습니다.")
        ),
    }


def calculate_fertilizer_amount(crop: str, area_m2: float, growth_stage: str) -> dict:
    """선택 과제용: 면적 기반 시비량(간단 추정)을 계산한다."""
    knowledge = load_knowledge()
    rate_table = knowledge.get("fertilizer_rate_per_10a", {})
    rate = rate_table.get(crop, {}).get(growth_stage)

    if rate is None:
        return {
            "found": False,
            "message": f"{crop}/{growth_stage} 시비량 기준이 없습니다.",
        }

    # 10a = 1,000㎡ 기준
    amount_kg = round(rate * (float(area_m2) / 1000.0), 2)
    return {
        "found": True,
        "crop": crop,
        "growth_stage": growth_stage,
        "area_m2": area_m2,
        "rate_kg_per_10a": rate,
        "estimated_amount_kg": amount_kg,
        "note": "교육용 간단 추정치이며, 실제 시비는 토양검정과 생육 상태를 반영해야 합니다.",
    }


AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "get_disease_info": get_disease_info,
    "get_fertilizer_recommend": get_fertilizer_recommend,
    "calculate_fertilizer_amount": calculate_fertilizer_amount,
}


# ---------------------------------------------------------------------------
# OpenAI tools 스키마
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "지정한 지역의 현재 날씨(기온, 습도, 강수확률)와 영농 유의 사항을 조회한다. 정식·방제·작업 가능 여부를 물을 때 사용한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "조회할 지역명. 예: 전주, 나주, 상주, 진주, 화성",
                    }
                },
                "required": ["region"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_disease_info",
            "description": "작물명과 증상으로 병해충 정보(병명, 원인, 방제 방법)를 조회한다. 잎 반점, 시들음, 흰가루 등 이상 증상 질문에 사용한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "crop": {
                        "type": "string",
                        "description": "작물명. 예: 고추, 토마토, 사과, 배추, 벼",
                    },
                    "symptom": {
                        "type": "string",
                        "description": "관찰된 증상. 예: 잎에 노란 반점, 시들음, 검은 반점",
                    },
                },
                "required": ["crop", "symptom"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_fertilizer_recommend",
            "description": "작물과 생육 단계에 맞는 비료 종류와 시비 가이드를 추천한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "crop": {
                        "type": "string",
                        "description": "작물명. 예: 고추, 배추, 벼, 사과, 토마토",
                    },
                    "growth_stage": {
                        "type": "string",
                        "description": "생육 단계. 예: 정식기, 생육기, 이삭패는시기, 수확기",
                    },
                },
                "required": ["crop", "growth_stage"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_fertilizer_amount",
            "description": "작물, 생육 단계, 재배 면적(㎡)을 받아 필요 시비량(kg)을 간단히 추정한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "crop": {"type": "string", "description": "작물명"},
                    "area_m2": {
                        "type": "number",
                        "description": "재배 면적(제곱미터)",
                    },
                    "growth_stage": {
                        "type": "string",
                        "description": "생육 단계. 예: 정식기, 생육기, 이삭패는시기",
                    },
                },
                "required": ["crop", "area_m2", "growth_stage"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Function Calling 대화 루프
# ---------------------------------------------------------------------------
def run_function_call(name: str, arguments_json: str) -> str:
    """LLM이 요청한 함수를 실행하고 JSON 문자열로 반환한다."""
    args = json.loads(arguments_json)
    func = AVAILABLE_FUNCTIONS.get(name)
    if func is None:
        return json.dumps({"error": f"알 수 없는 함수: {name}"}, ensure_ascii=False)
    result = func(**args)
    return json.dumps(result, ensure_ascii=False)


def chat_with_functions(client: OpenAI, messages: list[dict]) -> tuple[str, list[str]]:
    """
    Function Calling 1회 라운드(+필요 시 tool 결과 반영)를 수행한다.

    Returns:
        final_answer: 사용자에게 보여줄 최종 답변
        called_functions: 호출된 함수 요약 목록
    """
    called_functions: list[str] = []

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=0.3,
    )
    msg = response.choices[0].message

    # 함수 호출이 없으면 일반 답변
    if not msg.tool_calls:
        return (msg.content or "").strip(), called_functions

    # assistant tool_calls 메시지를 대화에 추가
    messages.append(
        {
            "role": "assistant",
            "content": msg.content,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        }
    )

    # 다중 함수 호출 지원
    for tool_call in msg.tool_calls:
        fn_name = tool_call.function.name
        fn_args = tool_call.function.arguments
        called_functions.append(f"{fn_name}({fn_args})")

        tool_result = run_function_call(fn_name, fn_args)
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": fn_name,
                "content": tool_result,
            }
        )

    # 함수 결과를 반영한 최종 자연어 답변 생성
    final_response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
    )
    final_answer = (final_response.choices[0].message.content or "").strip()
    return final_answer, called_functions


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------
def init_session() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []


def main() -> None:
    st.set_page_config(page_title="AI 영농 상담 챗봇", page_icon="🌾", layout="centered")
    st.title("AI 영농 상담 챗봇")
    st.caption("OpenAI Function Calling 기반 · Agent 프레임워크 미사용")

    init_session()

    if not API_KEY:
        st.error("OPENAI_API_KEY가 설정되지 않았습니다. `.env` 파일을 확인하세요.")
        st.stop()

    if not DATA_PATH.exists():
        st.error(f"지식 파일이 없습니다: {DATA_PATH.name}")
        st.stop()

    client = OpenAI(api_key=API_KEY)

    with st.sidebar:
        st.subheader("테스트 질문")
        st.markdown(
            """
- 광주 날씨 어때? 고추 정식해도 될까?
- 토마토 잎에 노란 반점이 생겼어
- 배추 생육기에 어떤 비료를 줘야 해?
- 나주 날씨 알려주고, 사과 검은 반점병 방제법도 알려줘
- 고추 생육기 기준으로 500㎡에 비료 얼마나 필요해?
- 안녕하세요?
            """
        )
        if st.button("대화 초기화"):
            st.session_state.messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            st.session_state.display_messages = []
            st.rerun()

    for item in st.session_state.display_messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("functions"):
                st.info("호출 함수\n\n" + "\n".join(f"- `{fn}`" for fn in item["functions"]))

    user_input = st.chat_input("영농 관련 질문을 입력하세요")
    if not user_input:
        return

    # 화면 표시용 사용자 메시지
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # API 대화 이력에 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("상담 내용을 준비하고 있습니다..."):
            try:
                answer, called = chat_with_functions(
                    client, st.session_state.messages
                )
            except Exception as exc:  # 수업 환경에서 원인 확인용
                answer = f"오류가 발생했습니다: {exc}"
                called = []

        st.markdown(answer)
        if called:
            st.info("호출 함수\n\n" + "\n".join(f"- `{fn}`" for fn in called))

    st.session_state.display_messages.append(
        {"role": "assistant", "content": answer, "functions": called}
    )
    # 최종 assistant 답변을 API 이력에도 남겨 다음 턴 맥락 유지
    st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
