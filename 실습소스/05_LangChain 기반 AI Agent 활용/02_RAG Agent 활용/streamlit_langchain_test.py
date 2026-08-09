import os
from dotenv import load_dotenv
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent
from langchain.tools import tool


# -------------------------------------------------------
# 0. 환경설정
# -------------------------------------------------------
load_dotenv("C:/env/.env")

st.set_page_config(page_title="LangChain AI Agent Web App", page_icon="🤖")
st.title("🤖 LangChain AI Agent Web App")
st.markdown("LangChain v1.0 + Streamlit 예제")

user_input = st.text_input("질문을 입력하세요:", "")

# -------------------------------------------------------
# 1. Tool 정의
# -------------------------------------------------------
@tool
def multiply(a: int, b: int) -> int:
    """두 수를 곱한 값을 반환한다."""
    return a * b

@tool
def get_weather(city: str) -> str:
    """도시 이름을 받아 예시용 날씨를 반환한다."""
    return f"{city}의 날씨는 맑음이다."

# -------------------------------------------------------
# 2. LLM 및 에이전트 구성
# -------------------------------------------------------
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

agent = create_agent(
    model=llm,
    tools=[multiply, get_weather],
    system_prompt=(
        "너는 사용자의 요청을 해결하기 위해 필요시 제공된 도구를 반드시 호출해야 하는 에이전트이다. "
        "도구를 직접 호출하지 않고 임의로 결과를 만들어내지 않는다."
    ),
)

# -------------------------------------------------------
# 3. 실행 및 출력
# -------------------------------------------------------
if st.button("에이전트 실행") and user_input.strip():
    with st.spinner("에이전트가 응답 중입니다..."):
        result = agent.invoke({
            "messages": [
                {"role": "user", "content": user_input}
            ]
        })
        st.subheader("결과:")
        st.write(result["messages"][-1].content)

# 사용 예시:
#  "부산의 날씨를 알려줘"
#  "2와 5를 곱한 결과는?"
#  "서울의 날씨와 3×7 결과를 알려줘"

# 윈도우 탐색기에서 C:/ 경로에 streamlit_langchain_test.py 파일을 복사  
# Anaconda Prompt 실행
# cd c:/
# streamlit run streamlit_langchain_test.py
