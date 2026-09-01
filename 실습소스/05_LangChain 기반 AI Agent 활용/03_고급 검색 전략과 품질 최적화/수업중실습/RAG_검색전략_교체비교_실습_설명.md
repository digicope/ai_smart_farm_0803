# RAG 검색 전략 교체 비교 실습 — 소스 구조와 기능 설명

대상 파일: `RAG_검색전략_교체비교_실습.ipynb`  
교과목: 고급 검색 전략과 품질 최적화  
과제: 동일한 RAG 시스템에서 검색 전략만 교체하고, 검색 결과·답변 품질·속도를 비교한다.

이 문서는 노트북의 **절 구성, 핵심 함수, 데이터 흐름, 비교 설계**를 설명한다. 코드를 처음부터 다시 쓰지 않고, 노트북을 읽고 실행할 때 길잡이로 쓰면 된다.

---

## 1. 한 줄 요약

농촌진흥청 연차보고서 PDF 3종을 Chroma에 인덱싱한 뒤, **같은 벡터스토어 · 같은 TOP_N · 같은 grounded 답변 체인** 위에서 Basic / HyDE / Multi-Query / Self-Query / Rerank만 바꿔 가며 결과를 비교하고, 멀티턴에서는 오래된 대화를 요약해 토큰을 줄인다.

핵심 설계 원칙은 **공정 비교**다. 임베딩 모델, 생성 모델, 최종 문서 수, 답변 프롬프트는 고정하고, 차이는 Retriever(검색 전략)와 Rerank의 1차 후보 수(`BASE_K`)뿐이다.

---

## 2. 관련 파일과 역할

```text
수업중실습/
├── RAG_검색전략_교체비교_실습.ipynb   ← 본 설명의 대상 (제출용 답안 노트북)
├── RAG_검색전략_교체비교_실습_설명.md ← 이 문서
├── 실습과제문제.txt
├── pdf_data/                          ← 연차보고서 PDF 3개
├── rag_engine.py                      ← 노트북과 동일한 엔진 (Streamlit용)
└── streamlit_rag_compare.py           ← 웹 UI 예제
```

| 파일 | 역할 |
|---|---|
| 노트북 | 인덱싱 → 전략 구현 → 비교표 → 멀티턴 압축까지 한 번에 실행 |
| `rag_engine.py` | 노트북 로직을 클래스로 분리. Streamlit이 import |
| `streamlit_rag_compare.py` | 단일 검색 / 전략 비교 / 멀티턴 / 가이드 탭 |

노트북이 `pdf_data/`를 찾지 못하면 같은 단원의 `실습과제/01_RAG 검색 전략 교체 비교 실습/pdf_data`를 재사용한다. Chroma persist 경로는 Windows 한글 경로 이슈를 피해 TEMP 아래 ASCII 폴더를 쓸 수 있다.

---

## 3. 전체 파이프라인

```text
PDF 3종
  → 페이지 로드 (PyMuPDF, 실패 시 PyPDF)
  → year / source / page 메타데이터 부여
  → RecursiveCharacterTextSplitter 청킹
  → Chroma Persistent (text-embedding-3-small)

질문
  → [전략별 Retriever] → Document 리스트 (길이 TOP_N)
  → format_docs()로 컨텍스트 문자열
  → grounded 프롬프트 + gpt-4o-mini
  → 출처 포함 답변
```

모든 전략의 입출력 형태는 같다.

```text
retrieve_*(question: str) -> list[Document]
generate_grounded_answer(question, docs) -> str
```

그래서 6절에서 `STRATEGIES` 딕셔너리만 순회하면 공정 비교가 된다.

---

## 4. 노트북 절 구성

노트북은 0~9절로 나뉜다. 과제 요구사항과 대응하면 다음과 같다.

| 절 | 제목 | 과제 항목 |
|---|---|---|
| 0 | 환경 설정 | API 키, 모델·TOP_N 고정 |
| 1 | 데이터 준비 | PDF 로드, 청킹, year 메타데이터 |
| 2 | Chroma 구축 | 로컬 Persistent 벡터 DB |
| 3 | 공통 답변 체인 | grounded 프롬프트, 생성 체인 고정 |
| 4 | 검색 전략 구현 | Basic / HyDE / Multi-Query / Self-Query / Rerank |
| 5 | 전략 레지스트리 | 비교용 `{이름: 함수}` 통일 |
| 6 | 성능 비교 | 동일 질문, 출처·연도·시간·답변 |
| 7 | 멀티턴 + 히스토리 압축 | 요약 vs 원문 토큰 비교 |
| 8 | 정리 | 상황별 추천 전략 |
| 9 | Streamlit | 웹 UI 실행 안내 |

실행 순서는 위에서부터 한 셀씩이다. 2절의 `vectorstore`, 3절의 `llm`/`answer_chain`, 4절의 `retrieve_*`가 이후 절의 전제다.

---

## 5. 0절 — 환경 설정

### 5.1 API 키

- 경로: `C:\env\.env`
- 필수: `OPENAI_API_KEY`
- 선택: `COHERE_API_KEY` (없으면 Cross-Encoder Rerank만 사용)
- 키 값은 print하지 않는다.

### 5.2 고정 파라미터

| 상수 | 기본값 | 의미 |
|---|---|---|
| `EMBEDDING_MODEL` | `text-embedding-3-small` | 인덱싱·검색 임베딩 |
| `LLM_MODEL` | `gpt-4o-mini` | HyDE, Multi-Query, Self-Query, 답변, 요약 |
| `CHUNK_SIZE` | 1000 | 청크 최대 문자 수 |
| `CHUNK_OVERLAP` | 150 | 청크 겹침 |
| `TOP_N` | 4 | 모든 전략이 최종 답변에 넣는 문서 수 |
| `BASE_K` | 12 | Rerank 1차 후보 수 (넓게 가져온 뒤 재점수화) |
| `FORCE_REINDEX` | `False` | `True`면 Chroma를 처음부터 다시 임베딩 |
| `COLLECTION_NAME` | `rda_annual_reports` | Chroma 컬렉션 이름 |

`temperature=0`으로 LLM을 만들어, 같은 질문에 대해 재작성·필터·답변이 재현 가능하게 한다.

### 5.3 경로 함수

| 함수 | 역할 |
|---|---|
| `resolve_notebook_dir()` | `pdf_data/*.pdf`가 있는 폴더를 cwd·상위에서 찾음 |
| `resolve_pdf_dir()` | 로컬 `pdf_data` 또는 실습과제 쪽 `pdf_data` |
| `resolve_persist_dir()` | ASCII면 `chroma_db/`, 아니면 TEMP/`rda_rag_chroma_<hash>` |

Windows에서 Chroma/SQLite가 한글 경로를 열지 못하는 경우를 우회한다.

---

## 6. 1절 — 데이터 준비

대상 PDF:

- `2022년도_농촌진흥사업_연차보고서.pdf`
- `2023년도_농촌진흥사업_연차보고서_내지(최종).pdf`
- `2024년도_농촌진흥사업_연차보고서.pdf`

### 6.1 함수

| 함수 | 역할 |
|---|---|
| `extract_year(filename)` | 파일명에서 `20xx` 연도를 정규식으로 추출 → 정수 |
| `_page_index(meta)` | `page` 또는 `page_number`를 정수로 정규화 (0-index) |
| `_load_pdf_pages(path)` | PyMuPDFLoader 우선, 실패 시 PyPDFLoader |
| `load_and_split_pdfs(...)` | 3개 PDF를 로드·필터·청킹하여 `Document` 리스트 반환 |

### 6.2 메타데이터

각 청크에 저장하는 필드:

| 키 | 값 |
|---|---|
| `source` | PDF 파일명 |
| `year` | 2022 / 2023 / 2024 (정수). Self-Query 필터에 사용 |
| `page` | 페이지 번호 (0-index) |

### 6.3 필터 기준

- 페이지 본문이 40자 미만이면 제외 (빈 페이지·목차 잔여물)
- 청크 본문이 80자 미만이면 제외
- 분리자: `\n\n` → `\n` → `. ` → 공백 → 문자 단위

2023 PDF는 서브셋 폰트/CID 때문에 추출 품질이 떨어질 수 있다. 그래서 PyMuPDF를 우선 쓰고, 연도 필터(Self-Query)가 텍스트 품질 이슈를 일부 보완한다.

---

## 7. 2절 — Chroma Vector DB

```text
OpenAIEmbeddings(text-embedding-3-small)
  → Chroma.from_documents(...)   # 최초 또는 FORCE_REINDEX
  또는
  → Chroma(persist_directory=...) # 기존 인덱스 재사용
```

이후 모든 Retriever는 **이 `vectorstore` 하나**만 본다. 전략마다 따로 임베딩하지 않는다.

`PERSIST_DIR`에 이미 파일이 있고 `FORCE_REINDEX=False`이면 임베딩 API 호출 없이 로드만 한다. 전체 PDF를 다시 인덱싱하면 수 분이 걸릴 수 있다.

---

## 8. 3절 — 공통 grounded 답변 체인

검색 전략이 달라도 **답변 생성은 한 체인**이다.

### 8.1 프롬프트 규칙

- 컨텍스트에 있는 내용만 근거로 답한다.
- 없으면 `"제공된 문서에서 답을 찾을 수 없습니다"`.
- 일반 지식·추측 금지.
- 마지막에 `(파일명, 연도, 페이지)` 출처를 붙인다.

잘못된 연도 문서를 가져오면 모델이 “모른다”고 답할 수 있다. 이는 실패가 아니라 **근거 없는 답을 막은 결과**로 해석한다.

### 8.2 함수

| 함수 | 역할 |
|---|---|
| `format_docs(docs)` | `[i] (source, year, page) + 본문` 문자열로 합침 |
| `generate_grounded_answer(q, docs)` | `answer_prompt \| llm \| StrOutputParser` 호출 |
| `preview_docs(docs)` | 출처·연도·페이지·앞 80자를 콘솔에 출력 |

LCEL 구성: `ChatPromptTemplate → ChatOpenAI → StrOutputParser`.

---

## 9. 4절 — 검색 전략

각 전략은 `question → List[Document]`이고, 최종 길이는 `TOP_N`이다. 샘플 질문(`SAMPLE_Q`)은 연도가 명시된 스마트농업 성과 질문이라, Self-Query 필터 효과를 바로 볼 수 있다.

### 9.1 Basic Retrieval (Naive)

```text
질문 임베딩 → similarity_search(k=TOP_N)
```

- 구현: `retrieve_basic()` → `vectorstore.similarity_search`
- 장점: 추가 LLM 호출 없음, 가장 빠르고 저렴
- 단점: 질문 문체가 보고서와 다르거나 연도 조건이 있으면 관련 없는 연도 청크가 섞일 수 있음

### 9.2 HyDE (Hypothetical Document Embeddings)

```text
질문 → LLM이 가상 답변 문단 생성 → 그 문서로 벡터 검색 → 실제 문서만 답변에 사용
```

| 단계 | 내용 |
|---|---|
| 1 | `hyde_chain`이 연차보고서 문체의 3~5문장 가상 문서를 생성 |
| 2 | `similarity_search(hypothetical_doc)` |
| 3 | 가상 문서는 **검색 쿼리 전용**. `generate_grounded_answer`의 컨텍스트에 넣지 않음 |

질문-문서 임베딩 간극(구어 vs 보고서 문체)을 줄이는 것이 목적이다. 비용은 LLM 1회 + 검색이다.

### 9.3 Multi-Query Retriever

```text
원 질문 → LLM이 여러 관점으로 재작성 → 각 쿼리로 검색 → unique union → TOP_N
```

- `langchain-classic`의 `MultiQueryRetriever.from_llm(..., include_original=True)`
- `_unique_union()`: `(source, page, 본문 앞 120자)` 키로 중복 제거
- 재현율(recall)을 올리지만 LLM 호출·검색 횟수가 늘어 느리다
- 한 질문에 하위 주제가 여러 개일 때(예: 청년농업인 + 강소농) 유리

### 9.4 Self-Query Retriever

```text
자연어 질문 → LLM이 (검색어, 메타데이터 필터)로 분리 → Chroma에서 필터 + 유사도 검색
```

- 필터 가능한 필드: `year`(integer), `source`(string), `page`(integer)
- `ChromaTranslator`로 구조화 쿼리를 Chroma 필터로 변환
- `"2024년 스마트농업 성과"` → 의미 검색어 + `year = 2024`
- 연차보고서처럼 연도가 중요한 코퍼스에서 다른 연도 노이즈를 줄인다
- 실행 시 `query_constructor`가 만든 구조화 쿼리를 출력해 필터가 걸렸는지 확인한다

### 9.5 Reranking

```text
1차: similarity_search(k=BASE_K=12)   # recall
2차: (질문, 문서) 쌍 재점수화 → 상위 TOP_N=4   # precision
```

**Cross-Encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`)

- 로컬 모델, API 키 불필요
- `retrieve_rerank_cross_encoder()`가 후보를 점수 내림차순 정렬

**Cohere Rerank** (`rerank-multilingual-v3.0`)

- `COHERE_API_KEY`가 있을 때만 `retrieve_rerank_cohere`를 정의
- 없으면 생략 (과제는 Cross-Encoder만으로 인정)

---

## 10. 5절 — 전략 레지스트리

```python
STRATEGIES = {
    "Basic": retrieve_basic,
    "HyDE": retrieve_hyde,
    "MultiQuery": retrieve_multi_query,
    "SelfQuery": retrieve_self_query,
    "Rerank(CrossEncoder)": retrieve_rerank_cross_encoder,
}
# Cohere 키가 있으면 Rerank(Cohere) 추가
```

6절 비교 루프는 이 딕셔너리만 순회한다. 새 전략을 넣으려면 같은 시그니처의 함수를 등록하면 된다.

---

## 11. 6절 — 성능 비교

### 11.1 비교용 질문과 의도

| 질문 | 의도 |
|---|---|
| 2024년 데이터 기반 스마트농업 확산과 고도화에서 어떤 성과가 있었나? | 연도 명시 → Self-Query `year=2024` 필터 효과 |
| 가루쌀 품종 바로미2의 가공 이용 연구는 어떻게 이루어졌나? | 문서 고유 용어 → Basic도 잘 맞을 수 있음. HyDE와의 표현 차이 |
| 청년농업인 육성과 스마트 강소농 지원은 어떻게 추진되었나? | 복합 주제 → Multi-Query recall |

### 11.2 `run_strategy()`가 기록하는 항목

| 필드 | 내용 |
|---|---|
| `strategy` | 전략 이름 |
| `years_hit` | 검색된 문서의 연도 집합 |
| `sources` | `파일명(y=연도, p=페이지)` 목록 |
| `retrieval_time_s` | 검색만 걸린 시간 |
| `generation_time_s` | 답변 생성 시간 |
| `total_time_s` | 합계 |
| `answer` / `answer_preview` | grounded 답변 전체 / 앞 180자 |

검색 시간과 생성 시간을 분리한 이유: HyDE·Multi-Query는 검색 단계에 LLM이 들어가고, 생성 시간은 컨텍스트 길이에 좌우된다.

### 11.3 해석 포인트 (정성)

골드셋이 없으므로 정확도는 **연도 적중**과 **출처가 질문 주제와 맞는지**로 본다.

- 질문 1에서 Basic/Multi-Query는 2022·2023 청크를 섞을 수 있다. Self-Query는 2024만 남기는 경우가 많다.
- 잘못된 연도 문맥이면 grounded 규칙 때문에 “답을 찾을 수 없습니다”가 나올 수 있다.
- 속도 대략: Basic < Rerank(로컬) < Self-Query / Multi-Query / HyDE (추가 LLM).

---

## 12. 7절 — 멀티턴 대화와 히스토리 압축

단원 `04_대화 히스토리 압축 및 컨텍스트 관리.ipynb`의 하이브리드 방식이다.

```text
총 입력 토큰 ≈ system + 대화히스토리 + RAG문서 + 현재질문
```

히스토리만 쌓이면 RAG 문서 자리가 줄어든다. 그래서 **히스토리 예산**과 **문서 예산**을 따로 센다 (`history_tokens`, `doc_tokens`, `prompt_tokens`).

### 12.1 `ConversationManager`

| 멤버 | 역할 |
|---|---|
| `turns` | `{question, answer}` 전체 턴 |
| `summary` | 오래된 턴의 누적 요약 |
| `keep_recent_turns` | 원문으로 남길 최근 턴 수 |
| `summarize_threshold` | 이 턴 수를 넘으면 압축 시작 |
| `_build_history_context()` | `[이전 대화 요약]` + 최근 턴 Q/A |
| `_maybe_compress()` | 최근 턴을 제외한 구간을 LLM으로 3~6문장 요약 |
| `ask(question)` | 검색 → grounded 답변(+히스토리) → 턴 저장 → 필요 시 압축 |

검색 전략은 Basic으로 고정한다. 압축 효과만 보기 위한 공정성이다.

### 12.2 비교 실험

| 조건 | 설정 | 의미 |
|---|---|---|
| 압축 없음 | `keep_recent_turns=100`, `summarize_threshold=10000` | 사실상 전체 원문 유지 |
| 압축 적용 | `keep_recent_turns=2`, `summarize_threshold=3` | 3턴 초과 시 오래된 턴 요약 |

7개 멀티턴 질문을 같은 순서로 넣고, 턴마다 프롬프트 토큰·히스토리 토큰·절감량을 표로 본다. 후반 턴일수록 `tokens_saved`가 커지는 것이 정상이다.

마지막에서 “앞서 설명한 스마트농업과 청년농업인 육성은 어떻게 연결되는가?”처럼 **이전 맥락을 참조하는 질문**으로, 압축 후에도 요지가 유지되는지를 확인한다.

---

## 13. 8절 — 상황별 추천 전략

| 상황 | 추천 | 이유 |
|---|---|---|
| 질문이 짧고 문서 용어와 비슷함 | Basic | 가장 빠르고 저렴 |
| 질문 문체가 보고서와 다름 | HyDE | 가상 문서로 임베딩 간극 축소 |
| 한 질문에 하위 주제가 여러 개 | Multi-Query | 재작성 합집합으로 recall |
| “2024년에는…”처럼 연도 조건 | Self-Query | year 필터로 다른 연도 배제 |
| 후보를 넓게 모은 뒤 상위 정확도 | Rerank | BASE_K → Cross-Encoder/Cohere |
| 대화가 길어지는 멀티턴 | 히스토리 압축 | 오래된 턴 요약 + 최근 원문 |

실무 파이프라인 예:

```text
Self-Query(연도 필터) → 넓은 k → Rerank(TOP_N) → grounded 생성
+ 멀티턴 요약 압축
```

---

## 14. 9절 — Streamlit 예제

노트북과 같은 `rag_engine.py`를 쓴다.

```bash
cd 수업중실습
streamlit run streamlit_rag_compare.py
```

| 탭 | 기능 |
|---|---|
| ① 단일 검색 | 전략 1개, 출처·답변·시간, HyDE 가상 문서 / Self-Query 필터 표시 |
| ② 전략 비교 | 같은 질문으로 여러 Retriever, 연도 적중·막대 차트 |
| ③ 멀티턴 대화 | 최근 원문 유지 턴 수·요약 임계값 조절, 토큰 추이 |
| ④ 전략 가이드 | 8절과 같은 추천 표 |

사이드바에서 `TOP_N`, `BASE_K`, 인덱스 재구축, 비교할 전략을 고른다. 비교 탭의 전략별 상세는 expander 안에 있으므로, HyDE 가상 문서 등은 expander를 중첩하지 않고 본문으로 그린다.

---

## 15. 주요 함수 목록 (노트북 기준)

### 경로·데이터

- `resolve_notebook_dir`, `resolve_pdf_dir`, `resolve_persist_dir`
- `extract_year`, `_page_index`, `_load_pdf_pages`, `load_and_split_pdfs`

### 생성

- `format_docs`, `generate_grounded_answer`, `preview_docs`

### 검색

- `retrieve_basic`
- `retrieve_hyde` (+ `hyde_chain`)
- `retrieve_multi_query` (+ `_unique_union`)
- `retrieve_self_query`
- `retrieve_rerank_cross_encoder`
- `retrieve_rerank_cohere` (키 있을 때만)

### 비교·대화

- `run_strategy`, `compare_strategies`
- `count_tokens`
- `ConversationManager.ask` / `_build_history_context` / `_maybe_compress`

---

## 16. 실행 시 주의

1. 셀은 0절부터 순서대로 실행한다. 중간부터 실행하면 `vectorstore`나 `STRATEGIES`가 없다.
2. 인덱스가 없으면 첫 임베딩에 OpenAI 호출과 시간이 든다. 이후에는 persist를 재사용한다.
3. 강제 재인덱싱은 `FORCE_REINDEX = True` 후 2절을 다시 실행한다.
4. Cohere 키가 없어도 Cross-Encoder만으로 과제 조건을 충족한다.
5. 비교 결과는 샘플링·모델 버전에 따라 조금씩 달라질 수 있다. 숫자의 절대값보다 **연도 필터, 속도 순서, grounded 여부** 패턴을 보면 된다.

---

## 17. 과제 요구사항 대응표

| 요구 | 노트북에서 하는 일 |
|---|---|
| PDF 전체 인덱싱, year 메타데이터 | 1절 `load_and_split_pdfs` |
| Chroma Persistent | 2절 |
| 임베딩·생성 모델 고정 | 0절 상수, 3절 체인 |
| Basic / HyDE / Multi-Query / Self-Query / Rerank | 4절 |
| 가상 문서는 검색용만 | `retrieve_hyde`가 실제 문서만 반환 |
| 멀티턴 + 오래된 턴 요약 | 7절 `ConversationManager` |
| 출처·연도·답변·속도 비교 | 6절 `comparison_df` |
| 압축 전후 토큰 비교 | 7절 `multi_turn_df` |
| 상황별 추천 | 8절 |
