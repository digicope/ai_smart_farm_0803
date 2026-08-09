
# agents_feedback_graph.py
# ==============================================================
# LangGraph 기반 피드백 루프형 워크플로우
# ==============================================================
import os
from dotenv import load_dotenv
from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableLambda
from langgraph.graph import StateGraph, END

# 환경 설정
load_dotenv("C:/env/.env")
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# --------------------------------------------------------------
# 상태 정의
# --------------------------------------------------------------
class AgentState(TypedDict):
    idea: str
    analysis: str
    content: str
    review: str
    feedback: str
    revised: str

# --------------------------------------------------------------
# Analyzer Node
# --------------------------------------------------------------
def analyzer_node(state: AgentState) -> AgentState:
    prompt = f"""
    다음 아이디어를 한 문장으로 요약하고, 핵심 키워드 3개를 추출하라.
    아이디어: {state['idea']}
    """
    result = llm.invoke(prompt)
    state["analysis"] = result.content.strip()
    return state

# --------------------------------------------------------------
# Writer Node (1차 작성)
# --------------------------------------------------------------
def writer_node(state: AgentState) -> AgentState:
    prompt = f"""
    아래 분석 내용을 바탕으로 3문장 이내의 홍보 문구를 작성하라:
    {state['analysis']}
    """
    result = llm.invoke(prompt)
    state["content"] = result.content.strip()
    return state

# --------------------------------------------------------------
# Reviewer Node (1차 평가)
# --------------------------------------------------------------
def reviewer_node(state: AgentState) -> AgentState:
    prompt = f"""
    다음 문구를 평가하라.
    - 명확성 (0~10)
    - 창의성 (0~10)
    - 문법 정확도 (0~10)
    - 개선 제안을 1문장으로 제시하라.
    문구:
    {state['content']}
    """
    result = llm.invoke(prompt)
    review_text = result.content.strip()
    state["review"] = review_text
    # 개선 제안만 추출
    state["feedback"] = review_text.split("\n")[-1]
    return state

# --------------------------------------------------------------
# Writer Node (피드백 반영 재작성)
# --------------------------------------------------------------
def rewriter_node(state: AgentState) -> AgentState:
    prompt = f"""
    다음 피드백을 반영하여 문구를 개선하라.
    기존 문구:
    {state['content']}

    피드백:
    {state['feedback']}
    """
    result = llm.invoke(prompt)
    state["revised"] = result.content.strip()
    return state

# --------------------------------------------------------------
# Reviewer Node (최종 평가)
# --------------------------------------------------------------
def final_reviewer_node(state: AgentState) -> AgentState:
    prompt = f"""
    아래 수정된 문구를 다시 평가하라.
    - 명확성 / 창의성 / 문법 정확도 점수를 다시 제시하라.
    문구:
    {state['revised']}
    """
    result = llm.invoke(prompt)
    state["review"] += "\n\n[최종 평가]\n" + result.content.strip()
    return state

# --------------------------------------------------------------
# LangGraph 구성 (피드백 루프 포함)
# --------------------------------------------------------------
def build_feedback_workflow():
    graph = StateGraph(AgentState)

    graph.add_node("analyzer", RunnableLambda(analyzer_node))
    graph.add_node("writer", RunnableLambda(writer_node))
    graph.add_node("reviewer", RunnableLambda(reviewer_node))
    graph.add_node("rewriter", RunnableLambda(rewriter_node))
    graph.add_node("final_reviewer", RunnableLambda(final_reviewer_node))

    # 흐름 연결
    graph.add_edge("analyzer", "writer")
    graph.add_edge("writer", "reviewer")
    graph.add_edge("reviewer", "rewriter")         # 리뷰 후 피드백 반영
    graph.add_edge("rewriter", "final_reviewer")   # 개선 후 최종 평가
    graph.set_entry_point("analyzer")
    graph.set_finish_point("final_reviewer")

    return graph.compile()

feedback_workflow = build_feedback_workflow()
