# AI 영농 상담 챗봇 답안 해설 

>  코드 답안은 `AI_영농_상담_챗봇_답안예제.py`를 참고한다.

## 1. 핵심 채점 포인트

| 항목 | 확인 내용 |
|---|---|
| Function Calling | `tools` 스키마 정의 및 `tool_choice="auto"` 사용 |
| Agent 미사용 | LangChain Agent / AutoGen 등 미사용 |
| 함수 구현 | `get_weather`, `get_disease_info`, `get_fertilizer_recommend` 포함 |
| 결과 피드백 | 함수 결과를 `role="tool"`로 다시 LLM에 전달 |
| UI | Streamlit 채팅으로 질문/답변 가능 |

## 2. 테스트 질문별 기대 동작

### (1) 전주 날씨 어때? 고추 정식해도 될까?

- 기대 함수: `get_weather(region="전주")`
- 기대 답변 포인트:
  - 기온 24°C, 습도 68%, 강수확률 20% 언급
  - 정식 가능하나 배수/소나기 주의 안내

### (2) 토마토 잎에 노란 반점이 생겼어

- 기대 함수: `get_disease_info(crop="토마토", symptom=...)`
- 기대 답변 포인트:
  - 잎곰팡이병(황화) 가능성
  - 환기, 이병엽 제거, 습도 관리, 등록 약제

### (3) 배추 생육기에 어떤 비료를 줘야 해?

- 기대 함수: `get_fertilizer_recommend(crop="배추", growth_stage="생육기")`
- 기대 답변 포인트:
  - 질소 추비 + 붕소 결핍 예방
  - 결구기 질소 과다 주의

### (4) 나주 날씨 알려주고, 사과 검은 반점병 방제법도 알려줘

- 기대 함수: `get_weather` + `get_disease_info` (다중 호출)
- 기대 답변 포인트:
  - 나주 날씨 요약
  - 검은별무늬병(흑성병) 방제 안내
  - 두 정보를 하나의 상담 문장으로 종합

### (5) 안녕하세요

- 기대 함수: 없음
- 기대 답변: 일반 인사 및 상담 가능 주제 안내

## 3. 확인 질문 모범 답안

### Q1. Function Calling을 쓰지 않고 LLM만으로 답하면 어떤 문제가 생기는가?

LLM이 최신/지역 날씨, 내부 지식 DB 값을 직접 조회하지 못해  
환각(없는 정보 생성)이 발생하기 쉽다.  
Function Calling은 검증 가능한 함수 결과를 기반으로 답하게 한다.

### Q2. `tool_choice="auto"`와 `"none"`의 차이는?

- `auto`: 모델이 필요 시 함수 호출 여부를 스스로 결정
- `none`: 함수를 호출하지 않고 텍스트만 생성

### Q3. 함수 실행 결과를 왜 다시 LLM에 전달해야 하는가?

함수 결과는 JSON/구조화 데이터인 경우가 많다.  
이를 다시 LLM에 전달해야 사용자에게 자연스럽고 상황 맞춤형 상담 문장으로 변환할 수 있다.

### Q4. 한 질문에서 날씨와 병해충을 동시에 물을 때 어떻게 처리하면 좋은가?

한 번의 응답에 여러 `tool_calls`가 올 수 있으므로  
각 함수를 실행한 뒤 모든 tool 결과를 메시지에 추가하고,  
최종 API 호출에서 종합 답변을 생성한다.

## 4. 실행 방법

```bash
pip install -r requirements.txt
streamlit run AI_영농_상담_챗봇_답안예제.py
```

제출 형식에 맞출 경우:

```bash
copy AI_영농_상담_챗봇_답안예제.py app.py
streamlit run app.py
```
