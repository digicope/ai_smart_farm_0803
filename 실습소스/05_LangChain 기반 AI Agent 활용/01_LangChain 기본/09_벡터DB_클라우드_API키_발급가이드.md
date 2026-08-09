# 클라우드 벡터 DB 회원가입 및 API Key 발급 가이드

`09_벡터 스토어.ipynb`의 Pinecone / Weaviate / Qdrant 예제를 실행하려면 아래 절차대로 각 서비스에 가입하고 API Key를 발급받아야 합니다.

세 서비스 모두 **클라우드에 벡터(임베딩)를 저장·검색**하는 매니지드 벡터 DB입니다. 로컬에 DB를 설치하지 않아도, API Key만 있으면 노트북에서 바로 연결해 실습할 수 있습니다.

## 사전 준비: `.env` 파일

발급받은 값은 노트북에서 `load_dotenv("C:/env/.env")`로 불러오므로, `C:/env/.env` 파일에 아래 형식으로 저장해 두면 됩니다.

```
PINECONE_API_KEY=발급받은_키
WEAVIATE_URL=https://xxxxxxx.weaviate.cloud
WEAVIATE_API_KEY=발급받은_키
QDRANT_URL=https://xxxxxxx.cloud.qdrant.io
QDRANT_API_KEY=발급받은_키
```

- 키/URL 앞뒤에 공백이나 따옴표(`"`)를 넣지 마세요. (`KEY="abc"` ❌ → `KEY=abc` ⭕)
- 파일 이름은 `.env`이고, 경로는 노트북의 `load_dotenv(...)` 인자와 일치해야 합니다.
- 실습에 쓰지 않는 서비스는 해당 줄을 비워 두거나 생략해도 됩니다. (해당 셀만 건너뛰면 됩니다.)

---

## 1. Pinecone

Pinecone은 서버리스 벡터 DB입니다. **API Key만** 있으면 되고, 클러스터 URL을 따로 받을 필요는 없습니다. (인덱스 생성/접속은 코드에서 Key로 처리)

1. [https://www.pinecone.io](https://www.pinecone.io) 접속 → **Sign Up Free** 클릭
2. Google/GitHub 계정 또는 이메일로 가입 (무료 **Starter** 플랜으로 진행 가능)
3. 로그인 후 왼쪽 메뉴에서 **API Keys** 클릭
   - 콘솔 UI가 바뀌었을 경우: 프로젝트 선택 후 **API Keys** / **Keys** 메뉴를 찾으면 됩니다.
4. 기본으로 생성되어 있는 `default` 키를 사용하거나 **Create API Key** 버튼으로 새 키 생성
5. 표시된 키 값을 복사 → `.env`의 `PINECONE_API_KEY`에 저장
   - 키는 보통 `pc-...` 형태로 시작합니다.

> **인덱스(index)** 는 노트북 코드에서 `pc.create_index(...)`로 자동 생성되므로 콘솔에서 미리 만들 필요는 없습니다.
>
> 무료 플랜은 리전이 제한적입니다. 노트북의 `ServerlessSpec(cloud="aws", region="us-east-1")` 값이 콘솔에서 사용 가능한 region과 다르면 인덱스 생성에 실패하므로, 콘솔의 허용 region을 확인한 뒤 코드의 `region`을 맞춰 주세요.
>
> 같은 이름의 인덱스가 이미 있으면 생성 단계에서 오류가 날 수 있습니다. 그 경우 콘솔에서 기존 인덱스를 삭제하거나, 코드의 인덱스 이름을 바꾸면 됩니다.

---

## 2. Weaviate

Weaviate Cloud는 **클러스터를 하나 만든 뒤**, 그 클러스터의 **URL + API Key** 두 가지가 모두 필요합니다.

1. [https://console.weaviate.cloud](https://console.weaviate.cloud) 접속 → **Sign Up** 클릭
2. Google/GitHub 계정 또는 이메일로 가입
3. 로그인 후 **Create Cluster** 클릭
4. 클러스터 타입은 **Free**(Sandbox) 선택 후 클러스터 이름 입력 → **Create cluster**
   - 생성에 1~2분 정도 걸릴 수 있습니다. 상태가 Ready/Running이 될 때까지 기다립니다.
5. 클러스터 생성이 끝나면 상세 화면에서:
   - **REST Endpoint** 값을 복사 → `.env`의 `WEAVIATE_URL`에 저장
     - 예: `https://xxxxxxx.weaviate.cloud` (`https://` 포함)
   - **Admin API Key** (또는 **API Keys** 탭)의 키 값을 복사 → `.env`의 `WEAVIATE_API_KEY`에 저장
     - 키가 없으면 **+ New key** 버튼을 눌러 새로 생성합니다.
     - 키는 보통 `wv_...` 또는 JWT처럼 긴 문자열 형태입니다.

> Free Sandbox는 **일정 기간(보통 14일 전후) 후 자동 삭제**되므로, 장기 사용 시 유료 플랜으로 전환하거나 만료 전 클러스터를 다시 만들어야 합니다.
>
> URL만 넣고 Key를 빠뜨리거나, 반대로 Key만 넣으면 인증 오류가 발생합니다. 두 값을 모두 `.env`에 넣었는지 확인하세요.

---

## 3. Qdrant

Qdrant Cloud도 Weaviate와 같이 **Cluster URL + API Key**가 필요합니다.

1. [https://cloud.qdrant.io](https://cloud.qdrant.io) 접속 → **Sign Up** 클릭
2. Google/GitHub 계정 또는 이메일로 가입
3. 로그인 후 **Clusters** 메뉴 → **Create Cluster** 클릭
4. **Free Tier** 선택 후 클러스터 이름/리전 입력 → **Create**
   - Free Tier는 클러스터 개수·용량에 제한이 있으니, 실습용으로 하나만 만들어 두면 충분합니다.
5. 클러스터 생성 완료 후:
   - 클러스터 상세 화면의 **Cluster URL**(Endpoint) 복사 → `.env`의 `QDRANT_URL`에 저장
     - 예: `https://xxxxxxx.cloud.qdrant.io` (`https://` 포함, 끝에 `/`는 보통 생략)
   - **API Keys** 탭 → **Create API Key** 클릭 → 표시된 키 복사 → `.env`의 `QDRANT_API_KEY`에 저장
     - 키는 발급 직후에만 전체 값이 보이는 경우가 많으니 바로 `.env`에 저장하세요.

> **컬렉션(collection)** 은 노트북 코드에서 `client.create_collection(...)`으로 자동 생성되므로 콘솔에서 미리 만들 필요는 없습니다.
>
> URL에 `https://`가 빠지거나 포트/경로가 잘못 붙으면 연결에 실패합니다. 콘솔에 표시된 Endpoint를 그대로 복사하는 것이 가장 안전합니다.

---

## 발급 후 확인 체크리스트

노트북을 실행하기 전에 아래를 한 번 확인하면 대부분의 연결 오류를 줄일 수 있습니다.

| 항목 | 확인 내용 |
|------|-----------|
| `.env` 경로 | `C:/env/.env` 파일이 실제로 존재하는지 |
| 변수 이름 | `PINECONE_API_KEY`, `WEAVIATE_URL` 등 철자가 정확한지 |
| URL 형식 | Weaviate/Qdrant URL에 `https://`가 포함되어 있는지 |
| 키 노출 | 노트북 셀에 키를 직접 붙여넣지 않았는지 |
| 재시작 | `.env`를 수정했다면 커널을 재시작한 뒤 `load_dotenv` 셀을 다시 실행했는지 |

간단한 확인 예 (노트북에서):

```python
import os
from dotenv import load_dotenv

load_dotenv("C:/env/.env")
print("Pinecone:", bool(os.getenv("PINECONE_API_KEY")))
print("Weaviate:", bool(os.getenv("WEAVIATE_URL") and os.getenv("WEAVIATE_API_KEY")))
print("Qdrant:", bool(os.getenv("QDRANT_URL") and os.getenv("QDRANT_API_KEY")))
```

`True`가 나오면 환경 변수 로드는 정상입니다. (`False`면 `.env` 경로·변수명을 다시 확인하세요.)

---

## 참고

- 세 서비스 모두 무료 플랜(Free/Starter/Sandbox)으로 예제 실습이 가능합니다.
- API Key는 발급 직후에만 전체 값이 표시되는 경우가 많으니, 발급 즉시 `.env` 파일에 붙여넣어 저장해 두는 것이 안전합니다.
- API Key는 절대 노트북/코드/GitHub에 직접 하드코딩하지 말고 `.env` 파일을 통해서만 사용합니다.
- 콘솔 UI 문구는 서비스 업데이트로 약간 달라질 수 있습니다. 메뉴 이름이 다르더라도 **API Key**, **Endpoint/URL**, **Create Cluster** 같은 핵심 항목을 찾으면 됩니다.
- 401/403(인증 실패)이 나오면 Key·URL이 잘못된 경우가 많고, 타임아웃/연결 거부는 URL·클러스터 상태(삭제·만료)를 먼저 확인하세요.
