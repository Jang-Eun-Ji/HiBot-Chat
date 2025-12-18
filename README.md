# HiBot-Chat (HiTalk / 하이톡) — 내부 문서 기반 사내 Q&A 챗봇

사내 규정·절차 문서가 많아 필요한 정보를 찾기 어렵고, 반복 문의가 발생하는 문제를 해결하기 위해 **내부 문서 기반 검색(RAG)** 과 **FAQ 표준화 자동응답**을 결합한 사내 챗봇입니다.

> **핵심**
> - FAQ는 **즉시 표준 답변**
> - FAQ에 없으면 내부 문서를 **Chunk → Embedding → DuckDB(색인카드)** 로 검색하고  
>   검색된 카드(근거) 범위 내에서만 **Gemini**가 답변을 생성합니다.

---

## 주요 기능

### 1) 문서 기반 검색 (Semantic Search + RAG)
- 내부 문서를 수집하여 작은 단락(Chunk)으로 분할
- 분할된 텍스트를 의미 벡터(Embedding)로 변환
- DuckDB에 저장된 색인카드(문서 chunk + embedding)를 기반으로 사용자 질문과 가장 유사한 내용을 검색
- FAQ에 없는 질문은 검색 결과(색인카드)와 질문을 조합하여 Gemini가 답변 생성  
  - 답변은 **색인카드 내용만 사용하도록 제어**하여 정확도/일관성 강화

### 2) FAQ 표준화 및 자동응답
- 미리 등록된 FAQ 질문을 버튼 형태로 제시
- 버튼 선택 시 해당 질문에 연결된 표준 답변을 즉시 출력

---

## 기술 스택

- **AI**: Gemini API
- **Backend**: FastAPI
- **DB**: DuckDB (`hibot_store.db`)
- **Frontend**: React (별도 `frontend/` 폴더)

---

## 레포 구조

```bash
HiBot-Chat/
├─ backend/
│  ├─ chatbot.py            # ✅ 핵심: FastAPI + RAG 질의응답 로직
│  ├─ build_index.py         # 문서 → chunk → embedding → DuckDB 색인 구축
│  ├─ hibot_store.db         # DuckDB (색인카드 저장소)
│  ├─ inspect_db.py          # DB 점검/확인용 스크립트
│  ├─ synonym_map.json       # 동의어/표현 보정(선택)
│  ├─ extract_text/          # 문서 텍스트 추출 관련 모듈/스크립트(선택)
│  ├─ requirements.txt
│  ├─ Procfile               # 배포용(플랫폼에 따라 사용)
│  └─ .env                   # (로컬) API Key 등 환경변수
└─ frontend/                 # React UI
