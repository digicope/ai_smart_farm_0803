
from dotenv import load_dotenv
import os

# .env 파일의 내용 불러오기
load_dotenv("C:/env/.env")

# 환경 변수 가져오기
API_KEY = os.getenv("OPENAI_API_KEY")

import streamlit as st

# ======================
# 1. OpenAI 클라이언트 초기화
# ======================
from openai import OpenAI
client = OpenAI(api_key=API_KEY)

# ======================
# 2. Streamlit 페이지 설정
# ======================
st.set_page_config(page_title="OpenAI Chatbot", page_icon="🤖", layout="centered")
st.title("🤖 OpenAI Chatbot with Streamlit")
st.markdown("간단한 **대화형 챗봇** 예제입니다. 모델 선택, temperature 조절, 대화 기록 저장 기능을 포함합니다.")

# ======================
# 3. 사이드바 옵션
# ======================
st.sidebar.header("⚙️ 설정")
model = st.sidebar.selectbox(
    "모델 선택",
    ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
    index=0
)
temperature = st.sidebar.slider("창의성 (temperature)", 0.0, 1.5, 0.7, 0.1)
max_tokens = st.sidebar.slider("최대 토큰 수", 50, 1000, 300, 50)

# ======================
# 4. 세션 상태 초기화 (대화 기록)
# ======================
if "messages" not in st.session_state:
    st.session_state.messages = []

# ======================
# 5. 기존 대화 기록 출력
# ======================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ======================
# 6. 사용자 입력
# ======================
if user_input := st.chat_input("질문을 입력하세요..."):
    # 사용자 메시지 저장
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # OpenAI API 호출
    with st.chat_message("assistant"):
        with st.spinner("🤔 답변 생성 중..."):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
                temperature=temperature,
                max_tokens=max_tokens
            )
            answer = response.choices[0].message.content
            st.markdown(answer)

    # 어시스턴트 응답 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})

    # 토큰 사용량 표시
    if hasattr(response, "usage"):
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"**🔢 사용된 토큰**")
        st.sidebar.markdown(f"- prompt_tokens: {response.usage.prompt_tokens}")
        st.sidebar.markdown(f"- completion_tokens: {response.usage.completion_tokens}")
        st.sidebar.markdown(f"- total_tokens: {response.usage.total_tokens}")

# 윈도우 탐색기에서 C:/ 경로에 streamlit_test.py 파일을 복사
# Anaconda Prompt 실행
# cd c:/
# streamlit run streamlit_test.py
