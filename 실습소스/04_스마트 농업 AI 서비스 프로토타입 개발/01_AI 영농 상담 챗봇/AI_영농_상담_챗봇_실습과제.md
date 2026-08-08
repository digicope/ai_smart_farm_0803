# [실습 과제] AI 영농 상담 챗봇 (Function Calling)

## 1. 실습 목표

OpenAI Function Calling을 활용하여 영농 상담용 챗봇을 구현한다.

사용자가 날씨, 병해충, 비료에 대해 질문하면 LLM이 적절한 함수를 선택하고,  
함수 실행 결과를 바탕으로 실용적인 영농 조언을 생성한다.

> 본 실습은 AI Agent 프레임워크를 사용하지 않는다.  
> OpenAI Chat Completions API의 **tools / Function Calling**만 사용한다.

## 2. 선수 학습

다음 내용을 학습한 후 진행한다.

- OpenAI API 기본 호출
- Function Calling (`tools`, `tool_choice`)
- Streamlit 기본 UI

## 3. 제공 자료

같은 폴더의 샘플 지식 데이터를 사용한다.

```text
farm_knowledge.json
```

데이터에는 다음 정보가 포함되어 있다.

- 지역별 날씨 더미 데이터
- 작물별 병해충 정보
- 작물별 비료 추천 정보

상위 폴더의 PDF 자료(채소/과수 병해충, 비료 사용법)는 참고용이다.  
본 과제에서는 `farm_knowledge.json`을 함수가 조회하는 방식으로 구현한다.

## 4. 구현할 기능

### (1) Function Calling용 함수 3개 정의

다음 3개 함수를 구현한다.

#### ① `get_weather`

지역 날씨 정보를 조회한다.

```text
입력: region (예: 전주, 나주, 상주)
출력: 기온, 습도, 강수확률, 농업 유의 사항
```

#### ② `get_disease_info`

작물과 증상으로 병해충 정보를 조회한다.

```text
입력: crop (예: 고추, 토마토, 사과), symptom (예: 잎에 반점)
출력: 병명, 원인, 방제 방법
```

#### ③ `get_fertilizer_recommend`

작물과 생육 단계에 맞는 비료를 추천한다.

```text
입력: crop (예: 배추, 벼), growth_stage (예: 정식기, 생육기, 수확기)
출력: 추천 비료, 시비량 가이드, 주의사항
```

### (2) tools 스키마 작성

OpenAI에 전달할 `tools` 리스트를 작성한다.

각 함수에 대해 다음을 포함한다.

- `name`
- `description` (언제 호출해야 하는지 LLM이 이해할 수 있게 작성)
- `parameters` (JSON Schema)

예:

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "지정한 지역의 현재 날씨 및 영농 유의 정보를 조회한다.",
            "parameters": {
                "type": "object",
                "properties": {
                    "region": {
                        "type": "string",
                        "description": "조회할 지역명. 예: 전주, 나주"
                    }
                },
                "required": ["region"]
            }
        }
    }
]
```

### (3) Function Calling 대화 루프 구현

다음 흐름으로 챗봇을 구현한다.

```text
1. 사용자 질문 입력
2. OpenAI API 호출 (tools 포함)
3. tool_calls가 있으면
   - 함수 이름/인자 추출
   - 실제 파이썬 함수 실행
   - 함수 결과를 role="tool" 메시지로 추가
   - 다시 OpenAI API 호출하여 최종 답변 생성
4. tool_calls가 없으면
   - 일반 텍스트 답변 출력
```

핵심 구조 예:

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
    tool_choice="auto",
)

msg = response.choices[0].message

if msg.tool_calls:
    # 함수 실행 → tool 메시지 추가 → 최종 답변 생성
    ...
else:
    # 일반 답변
    ...
```

### (4) Streamlit 채팅 UI

Streamlit을 이용하여 간단한 상담 화면을 구성한다.

- 제목: `AI 영농 상담 챗봇`
- 채팅 입력창
- 사용자/챗봇 메시지 표시
- (선택) 어떤 함수가 호출되었는지 표시

## 5. 테스트 질문 예시

다음 질문으로 Function Calling이 동작하는지 확인한다.

| 질문 | 기대 호출 함수 |
|---|---|
| 전주 날씨 어때? 고추 정식해도 될까? | `get_weather` |
| 토마토 잎에 노란 반점이 생겼어 | `get_disease_info` |
| 배추 생육기에 어떤 비료를 줘야 해? | `get_fertilizer_recommend` |
| 나주 날씨 알려주고, 사과 검은 반점병 방제법도 알려줘 | `get_weather` + `get_disease_info` |
| 안녕하세요 | 함수 호출 없음 (일반 인사) |

## 6. 최종 화면 예시

```text
==================================================
           AI 영농 상담 챗봇
==================================================

[사용자]
전주 날씨 어때? 고추 정식해도 될까?

[시스템]
호출 함수: get_weather(region="전주")

[챗봇]
전주 현재 기온은 24°C, 습도 68%, 강수확률 20%입니다.
기온과 습도 조건이 양호하여 고추 정식에 큰 무리는 없습니다.
다만 오후 소나기 가능성이 있으니 정식 후 물빠짐을 확인해 주세요.

==================================================
```

## 7. 구현 조건

### 필수

1. AI Agent 라이브러리(LangChain Agent, AutoGen 등)를 사용하지 않는다.
2. OpenAI `tools` 기반 Function Calling을 사용한다.
3. 함수는 최소 3개 이상 구현한다.
4. 함수 결과를 다시 LLM에 전달하여 최종 상담 문장을 생성한다.
5. Streamlit 채팅 UI로 실행 가능해야 한다.

### 권장

- 시스템 프롬프트에 "영농 상담 전문가" 역할을 명시한다.
- 지식 데이터에 없는 내용은 추측하지 말고, 확인이 필요하다고 안내한다.
- 함수 호출 여부를 화면에 간단히 표시한다.

## 8. 선택 과제

기본 기능을 완성한 경우 다음 중 하나를 추가한다.

### 선택 과제 1. 다중 함수 호출

한 질문에서 여러 함수를 연속 호출하고 결과를 종합 답변한다.

예:

```text
상주 날씨 알려주고, 벼 이삭 패는 시기에 비료 추천도 해줘
```

### 선택 과제 2. 시비량 계산 함수 추가

```text
calculate_fertilizer_amount(crop, area_m2, growth_stage)
```

재배 면적을 받아 필요 시비량을 계산한다.

### 선택 과제 3. 대화 기록 유지

Streamlit `st.session_state`에 대화 이력을 저장하여  
이전 맥락을 이어가는 상담이 가능하게 한다.

## 9. 실행 환경

필요한 라이브러리를 설치한다.

```bash
pip install openai python-dotenv streamlit
```

`.env` 파일에 API 키를 설정한다.

```text
OPENAI_API_KEY=sk-...
```

프로그램을 다음 명령으로 실행한다.

```bash
streamlit run app.py
```

## 10. 제출 파일

```text
AI_영농_상담_챗봇/
│
├── app.py
├── farm_knowledge.json
├── 출력화면캡쳐.jpg
└── requirements.txt   (선택)
```

## 11. 실습 시간

총 실습 시간: 1시간 30분

| 시간 | 내용 |
|---|---|
| 10분 | `farm_knowledge.json` 구조 확인 |
| 20분 | 함수 3개 및 tools 스키마 작성 |
| 30분 | Function Calling 대화 루프 구현 |
| 20분 | Streamlit 채팅 UI 연결 |
| 10분 | 테스트 질문 검증 및 캡처 |

## 12. 확인 질문

구현 후 다음 질문에 답한다.

1. Function Calling을 쓰지 않고 LLM만으로 답하면 어떤 문제가 생기는가?
2. `tool_choice="auto"`와 `"none"`의 차이는 무엇인가?
3. 함수 실행 결과를 왜 다시 LLM에 전달해야 하는가?
4. 한 질문에서 날씨와 병해충을 동시에 물을 때 API 응답은 어떻게 처리하면 좋은가?
