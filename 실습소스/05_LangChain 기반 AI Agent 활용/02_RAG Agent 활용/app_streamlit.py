
# app_streamlit.py
import streamlit as st
from agents_feedback_graph import feedback_workflow

# 페이지 설정
st.set_page_config(page_title="LangGraph 피드백 루프 에이전트", layout="centered")

# ------------------------------------------------------------
# ✨ 커스텀 CSS (폰트 크기, 중앙 정렬, 줄간격 개선)
# ------------------------------------------------------------
st.markdown("""
<style>
    /* 제목 전체 정렬 */
    .main-title {
        text-align: center;
        font-size: 2rem;             /* 기존보다 작게 */
        font-weight: 700;
        margin-bottom: 0.2em;
        line-height: 1.2;
    }

    /* 부제목(워크플로우 단계 표시) */
    .subtitle {
        text-align: center;
        font-size: 0.9rem;
        color: #6c757d;
        margin-bottom: 1.5em;
    }

    /* 입력 안내 문구 */
    label[data-testid="stTextAreaLabel"] {
        font-weight: 600;
        font-size: 1rem;
    }

    /* 버튼 가운데 정렬 */
    div.stButton > button {
        display: block;
        margin: 0 auto;
        width: 180px;
    }

    /* 결과 영역 개선 */
    .stSuccess {
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# 제목 + 부제목
# ------------------------------------------------------------
st.markdown('<h1 class="main-title">🔄 LangGraph 피드백 루프 기반 AI 에이전트</h1>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Analyzer → Writer → Reviewer → Rewriter → Final Reviewer 순서로 자동 실행</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 입력 영역
# ------------------------------------------------------------
user_input = st.text_area("💡 아이디어를 입력하세요:", height=100)

if st.button("워크플로우 실행"):
    if not user_input.strip():
        st.warning("아이디어를 입력해주세요.")
    else:
        with st.spinner("에이전트가 협업 중입니다..."):
            state = {"idea": user_input}

            for event in feedback_workflow.stream(state):
                if isinstance(event, tuple) and len(event) == 2:
                    step, output = event
                else:
                    step, output = "unknown", event

                if step == "analyzer":
                    st.markdown("### 🔍 Step 1: Analyzer 결과")
                    st.write(output.get("analysis", "결과 없음"))
                elif step == "writer":
                    st.markdown("### ✍️ Step 2: Writer (1차 문구)")
                    st.write(output.get("content", "결과 없음"))
                elif step == "reviewer":
                    st.markdown("### 🧾 Step 3: Reviewer (1차 평가)")
                    st.write(output.get("review", "결과 없음"))
                elif step == "rewriter":
                    st.markdown("### ♻️ Step 4: Rewriter (피드백 반영)")
                    st.write(output.get("revised", "결과 없음"))
                elif step == "final_reviewer":
                    st.markdown("### ✅ Step 5: Final Reviewer (최종 평가)")
                    st.write(output.get("review", "결과 없음"))

            # 최종 결과 출력
            st.divider()
            st.markdown("## 🏁 최종 결과 요약")
            result = feedback_workflow.invoke(state)
            st.write(result.get("revised", "최종 수정된 문구 없음"))
            st.write(result.get("review", "최종 평가 없음"))

            st.success("🎉 워크플로우 완료!")
