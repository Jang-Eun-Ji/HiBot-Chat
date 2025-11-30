import os
import duckdb
import json
import numpy as np
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import Document
import google.generativeai as genai
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware


# --- 2. 경로 및 모델 설정 ---
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # build_index.py와 동일한 모델 사용
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"  # 한국어 모델 (SSL 문제 해결 후 사용)
DB_PATH = "hibot_store.db"  # build_index.py와 동일한 DuckDB 파일 경로
# KEYWORD_FILE = "document_keywords.json" # 문서 키워드 매핑 파일 경로
SYNONYM_MAP_PATH = "synonym_map.json" # 동의어 파일 경로 
EMPLOYEE_JSON_PATH = "employee_roles.json" # 직원 역할 정보 파일 경로

text_embedder = None
retriever = None
prompt_builder = None



# --- 0. [필수] API 키 설정 ---
# .env 파일에서 환경변수 로드
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
else:
    print("⚠️  경고: GOOGLE_API_KEY가 설정되지 않았습니다.")
    # (API 키가 없어도 FAQ 기능은 작동합니다)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 모든 도메인 허용
    allow_credentials=True,  
    allow_methods=["*"],      # 모든 HTTP 메서드 허용
    allow_headers=["*"],      # 모든 헤더 허용
)

# --- 1. [신규] 규칙 기반 FAQ 데이터베이스 (Req 1 & 2) ---
# 기획안의 "Quick Reply" 및 "FAQ 자동 응답"용
# 키(Keyword)가 질문에 포함되어 있으면, AI(RAG)를 호출하지 않고 즉시 이 답변을 반환합니다.
# (키워드를 구체적으로 적을수록 좋습니다)
# --- 1. 순서 기반 FAQ 리스트로 변경 ---
FIXED_FAQ_DATABASE = [
    "[인사근태(확장)] → [시간외근무] → [시간외근무 신청관리] 메뉴에서 신청 가능합니다. 1일 최대 3시간 30분, 월 최대 15시간까지 신청 가능하며, 휴게시간 30분을 제외해 입력해야 합니다. \n\n📄 출처: [컨텐츠1] 경영지원 업무 질의응답(Q&A)_202509_공개용 ",
    "급여 담당자 이메일로 가족수당 신청서와 증빙서류(가족관계증명서, 건강보험 자격확인서 등)를 제출하면 됩니다. 배우자 4만원, 직계존속·비속 각 3만원이 지급됩니다. ※ 관련근거: 보수규정 시행규칙 별표 제1호 \n\n📄 출처: [컨텐츠1] 경영지원 업무 질의응답(Q&A)_202509_공개용",
    "정규직·계약직 직원에게 연간 1,000,000포인트(1P=1원)가 부여되며, 단체보험료 공제 후 잔액 한도 내에서 사용 가능합니다. 입·퇴사자는 근무기간에 따라 월할 계산 적용됩니다. \n\n📄 출처: [컨텐츠1] 경영지원 업무 질의응답(Q&A)_202509_공개용",
    "출장신청은 [인사근태(확장)] → [근태신청서] → [출장신청] 메뉴에서 가능합니다. 국내출장은 1주일 이내, 국외출장은 2주일 이내에 운임·숙박비 등 증빙서류를 첨부하여 정산해야 합니다. \n\n📄 출처: [컨텐츠1] 경영지원 업무 질의응답(Q&A)_202509_공개용",
    "전산기기 및 사무기기(PC, 복합기, 세단기 등)는 기기 중간 또는 하단에 부착된 수리기사 연락처로 직접 유선 문의하시면 됩니다. 기타 시설물(조명, 의자, 문손잡이 등) 고장은 경영지원부 물품관리 담당자에게 연락해 주시기 바랍니다.\n\n📄 출처: [컨텐츠1] 경영지원 업무 질의응답(Q&A)_202509_공개용"
]

FAQ_KEYWORDS = [
    ["시간외근무", "시간 외 근무", "연장근무"],
    ["가족수당", "가족 수당"],
    ["복지포인트", "복지 포인트"],
    ["출장", "여비정산", "정산"],
    ["전산장비", "PC", "프린터", "시설물", "고장"]
]

# employee_roles.json 로딩 함수
def load_employee_roles():
    try:
        with open(EMPLOYEE_JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ employee_roles.json 불러오기 실패: {e}")
        return []

EMPLOYEES = load_employee_roles()

def find_best_employee(question: str):
    """
    질문을 기반으로 가장 관련 있는 직원 추천
    매칭 점수 기준:
    - 질문 키워드가 업무(task)에 등장하면 +1
    """
    if not EMPLOYEES:
        return None

    # 질문을 단어로 분리
    keywords = [w for w in question.split() if len(w) >= 2]

    best_match = None
    best_score = 0

    for emp in EMPLOYEES:
        score = 0
        for task in emp["tasks"]:
            for kw in keywords:
                if kw in task:
                    score += 1

        if score > best_score:
            best_score = score
            best_match = emp

    return best_match


# 동의어 맵 로드 함수
def load_synonym_map():
    try:
        with open(SYNONYM_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ synonym_map.json 불러오기 실패: {e}")
        return {}

SYNONYM_MAP = load_synonym_map()

# 긴 문서를 문장 단위로 자르는 함수
# def smart_trim(text, max_length=600):
#     if not text:
#         return ""

#     if len(text) <= max_length:
#         return text

#     trimmed = text[:max_length]

#     # 여러 후보 문장부호 검색
#     end_marks = ['다.', '요.', '함.', '.', '!', '?', '\n']

#     last_cut = -1
#     for mark in end_marks:
#         pos = trimmed.rfind(mark)
#         if pos != -1:
#             end_pos = pos + len(mark)
#             if end_pos > last_cut:
#                 last_cut = end_pos

#     # 문장부호 찾은 경우
#     if last_cut != -1:
#         return trimmed[:last_cut]

#     # 문장부호 없으면 단어 기준으로 자름
#     last_space = trimmed.rfind(" ")
#     if last_space != -1:
#         return trimmed[:last_space]

#     return trimmed



# --- 3. 질문과 비슷한 문서를 DuckDB에서 찾아주는 검색 엔진 ---
class DuckDBEmbeddingRetriever:
    """
    DuckDB 기반 semantic search retriever
    ✔ top_k 개수 제한
    ✔ similarity threshold 적용
    ✔ similarity 정보 meta에 저장
    ✔ 예쁘게 로그 출력
    """

    def __init__(self, db_path, top_k=6, threshold=0.5):
        self.db_path = db_path
        self.top_k = top_k
        self.threshold = threshold
        self.conn = None
        
    def connect(self):
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)


    def run(self, query_embedding):
        """query_embedding(list) → DuckDB에서 문서 리스트 반환"""
        self.connect()

        # 모든 문서 로드
        docs_data = self.conn.execute("""
            SELECT id, content, meta, embedding 
            FROM documents 
            WHERE embedding IS NOT NULL
        """).fetchall()

        if not docs_data:
            return {"documents": []}

        query_emb = np.array(query_embedding[0])
        similarities = []

        print("\n📘 [DuckDB Retriever] 문서 유사도 계산 시작")
        print(f" - Threshold = {self.threshold}")
        print(" - --------------------------------------------")

        # 각 문서와 similarity 계산
        for doc_id, content, meta_str, embedding in docs_data:
            if not embedding:
                continue

            doc_emb = np.array(embedding)

            # 코사인 유사도 계산
            similarity = float(
                np.dot(query_emb, doc_emb) 
                / (np.linalg.norm(query_emb) * np.linalg.norm(doc_emb))
            )

            # 메타 로드
            try:
                meta = json.loads(meta_str) if meta_str else {}
            except:
                meta = {}

            file_name = meta.get("file_name", "알 수 없음")
            page = meta.get("page_number", "N/A")

            # 예쁜 로그 출력
            print(f"🔎 문서: {file_name} (p.{page}) → 유사도: {similarity:.4f}")

            # threshold 미달 → 건너뛰기
            if similarity < self.threshold:
                continue

            similarities.append((similarity, doc_id, content, meta))

        print(" - --------------------------------------------")

        # threshold 미달 문서만 있었다면
        if not similarities:
            print("❌ threshold 이상 문서 없음 → 문서 없음으로 처리됨")
            return {"documents": []}

        # 상위 top_k만 선택
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_docs = similarities[:self.top_k]

        # Document 객체 생성
        documents = []
        print("\n📘 최종 선택된 문서(top_k)")
        for similarity, doc_id, content, meta in top_docs:
            meta["similarity"] = similarity
            print(f"✔ {meta.get('file_name', '알 수 없음')} → {similarity:.4f}")
            documents.append(Document(id=doc_id, content=content, meta=meta))

        print("------------------------------------------------\n")

        return {"documents": documents}

    

def find_representative_keyword(question: str):
    """
    사용자의 질문에 SYNONYM_MAP의 동의어가 포함되어 있으면 
    대표 키워드를 반환하는 함수
    예: '야근 신청 어떻게?' → '시간외근무'
    """
    for rep_keyword, synonyms in SYNONYM_MAP.items():
        # 대표 단어 자체가 질문에 있는 경우
        if rep_keyword in question:
            return rep_keyword
        
        # 동의어들이 질문 안에 포함되어 있는지
        for syn in synonyms:
            if syn in question:
                return rep_keyword

    return None



# --- 4. [신규] RAG 파이프라인 "라우터" (Req 3) ---

def initialize_chatbot():
    print("챗봇 초기화 중...")
    
    # (A) DuckDB 연결 확인
    try:
        if not os.path.exists(DB_PATH):
            print(f"❌ '{DB_PATH}' 데이터베이스 파일을 찾을 수 없습니다.")
            print("먼저 'python build_index.py' 스크립트를 실행하여 문서를 색인해주세요.")
            return None
        
        conn = duckdb.connect(DB_PATH)
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        conn.close()
        print(f"✅ '{DB_PATH}'에서 {doc_count}개 문서를 확인했습니다.")
    except Exception as e:
        print(f"❌ '{DB_PATH}' 데이터베이스 연결 실패: {e}")
        print("먼저 'python build_index.py' 스크립트를 실행하여 문서를 색인해주세요.")
        return None

    # (B) RAG 파이프라인 준비 (SSL 오류 처리 포함)
    try:
        # 한국어 문장을 숫자 벡터로 자동 변환해주는 모델을 로딩 
        text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)
        # 임베딩 기반 검색기(semantic search engine)
        # DuckDB 파일(hibot_store.db)에 접속해서 문서들의 임베딩(vector) 목록을 읽고 
        # 질문의  임베딩과 코사인 유사도(similarity score)를 계산해서 가장 비슷한 문서 **12(top_k=12)**를 반환함
        retriever = DuckDBEmbeddingRetriever(db_path=DB_PATH, top_k=6)
        print("✅ 임베더와 리트리버 초기화 완료")
    except Exception as e:
        print(f"❌ 임베더 초기화 실패: {e}")
        print("📋 해결방법:")
        print("   1. pip install --upgrade certifi")
        print("   2. 인터넷 연결 확인")
        return None
    
    prompt_template = """
당신은 내부 규정·지침·업무 매뉴얼을 기반으로 답변하는 AI 어시스턴트입니다.

아래 [참고 문서]는 질문과 가장 연관성이 높은 문서들입니다. 질문자가 이해하기 쉽게 문서를 [참고 문서]를 정리 해서 답변해 주세요.

[참고 문서]
{% for doc in documents %}
문서 {{ loop.index }}:
- 파일명: {{ doc.meta.file_name }}
- 유사도: {{ doc.meta.similarity }}
- 내용:
{{ doc.content }}
{% endfor %}

[질문]: {{ question }}

[답변]:
"""
    prompt_builder = PromptBuilder(template=prompt_template, required_variables=["documents", "question"])
    
    # (C) 임베더 초기화 (SSL 오류 처리)
    try:
        # 임베더 초기화 (SSL 오류 처리)
        text_embedder.warm_up()
        print("✅ 챗봇 RAG 파이프라인 준비 완료.")
        return text_embedder, retriever, prompt_builder
    except Exception as e:
        print(f"❌ 파이프라인 초기화 실패: {e}")
        return None

def create_gemini_response(prompt):
    """Gemini API를 직접 사용하여 응답을 생성하는 함수 """
    try:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-2.0-flash')  # Updated to available model
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        error_msg = str(e)

        # 1) AI 무료 사용량(Quota) 초과 또는 Rate Limit 초과
        if "429" in error_msg or "Resource exhausted" in error_msg:
            raise HTTPException(
                status_code=429,
                detail="AI Quota Exceeded"
            )

        # 2) API Key 문제
        if "API key" in error_msg or "permission" in error_msg.lower():
            raise HTTPException(
                status_code=403,
                detail="Permission Denied"
            )

        # 3) 기타 오류
        raise HTTPException(
            status_code=500,
            detail="Gemini Internal Error"
        )
    
# --- 5. 백엔드 테스트용 챗봇 실행 ---
# def ask_chatbot(question, text_embedder, retriever, prompt_builder):
    """
    (✨ 신규 로직)
    사용자 질문을 받아서 FAQ(규칙)를 먼저 확인하고, 
    없으면 RAG 파이프라인을 실행하는 메인 "라우터"
    """
    print(f"\n[질문] 💬: {question}")
    
    # --- 1단계: 규칙 기반 FAQ 확인 (Req 1 & 2) ---
    # 기획안의 "키워드 포함 여부" 로직
    for idx, keywords in enumerate(FAQ_KEYWORDS):
        for kw in keywords:
            if kw in question:
                return FIXED_FAQ_DATABASE[idx]
            
    # 2-A) 먼저 동의어 기반 대표 키워드 매핑
    rep_keyword = find_representative_keyword(question)
    if rep_keyword:
        print(f"🔍 동의어 매핑: '{question}' → 대표 키워드 '{rep_keyword}'로 검색")
        question = rep_keyword

    # --- 2단계: RAG + LLM 응답 (Req 3) ---
    print("(규칙 기반 답변 없음. RAG 파이프라인 실행...)")
    try:
        # (A) 질문을 임베딩으로 변환
        query_embedding_result = text_embedder.run(text=question)
        # retriever가 읽을 수 있도록 임베딩만 꺼내는 작업
        query_embedding = query_embedding_result["embedding"]
        
        # (B) 관련 문서 검색
        retrieved_docs = retriever.run(query_embedding=[query_embedding])["documents"]
        
        if not retrieved_docs:
            emp = find_best_employee(question)

            if emp:
                dept = emp["department"]
                name = emp["name"]
                pos = emp["position"]
                phone = emp["phone"]

                return {
                    "response": (
                        f"해당 질문에 대해서는 정확한 안내가 어렵습니다.\n"
                        f"자세한 내용은 {dept} {name} {pos}님({phone})께 문의 부탁드립니다."
                    )
                }

            return {
                "response": (
                    "해당 질문에 대해 관련 문서와 담당자를 찾을 수 없습니다.\n"
                    "경영지원부로 문의 부탁드립니다."
                )
            }

        
        prompt_docs = []
        for d in retrieved_docs:
            prompt_docs.append(
                Document(id=d.id, content=d.content, meta=d.meta)
    )

        prompt_result = prompt_builder.run(documents=prompt_docs, question=question)

        full_prompt = prompt_result["prompt"]
        
        # (D) Gemini API로 답변 생성
        answer = create_gemini_response(full_prompt)
        print(f"[답변] 🤖 (AI 생성): {answer}")
        return answer
        
    except Exception as e:
        error_msg = f"챗봇 실행 중 오류 발생: {str(e)}"
        print(f"[오류] ❌: {error_msg}")
        return error_msg


# if __name__ == "__main__":
#     # 챗봇 파이프라인 1회 초기화
#     pipeline_components = initialize_chatbot()
    
#     if pipeline_components:
#         text_embedder, retriever, prompt_builder = pipeline_components
        
#         # (테스트)
        
#         # (1) FAQ 질문 (RAG 미사용)
#         ask_chatbot("연차 어떻게 사용하나요?", text_embedder, retriever, prompt_builder)
        
#         # (2) 문서 기반 질문 (RAG 사용)
#         ask_chatbot("정보공개를 청구받은 부서는 며칠 내에 처리 해야해?", text_embedder, retriever, prompt_builder)

# 챗봇 부팅 로직
@app.on_event("startup")
def startup_event():
    global text_embedder, retriever, prompt_builder
    pipeline_components = initialize_chatbot()
    if pipeline_components:
        text_embedder, retriever, prompt_builder = pipeline_components


@app.post("/api/chat")
async def chat(request: Request):
    global text_embedder, retriever, prompt_builder
    data = await request.json()
    question = data.get("message", "")
    print(f"💬 사용자 질문: {question}")

    # 1️⃣ 규칙 기반 FAQ 먼저 확인
    for idx, keywords in enumerate(FAQ_KEYWORDS):
        for kw in keywords:
            if kw in question:
                return {"response": FIXED_FAQ_DATABASE[idx]}


    # 2️⃣ RAG + Gemini 호출
    
    rep_keyword = find_representative_keyword(question)
    if rep_keyword:
        print(f"🔍 동의어 매핑: '{question}' → '{rep_keyword}'")
        question = rep_keyword

    # 3️⃣ 질문 임베딩 생성 
    query_emb = text_embedder.run(text=question)["embedding"]
    # 4️⃣ DuckDB 검색
    docs = retriever.run(query_embedding=[query_emb])["documents"]
    print(f"DuckDB 검색된 문서 개수: {len(docs)}")

    if not docs:
        # 관련 문서 없으면 담당자 추천
        print("🔍 관련 문서 없음 → 담당자 추천 로직 실행")
        emp = find_best_employee(question)

        if emp:
            dept = emp["department"]
            name = emp["name"]
            pos = emp["position"]
            phone = emp["phone"]

            return {
                "response": (
                    f"해당 질문에 대해서는 정확한 안내가 어렵습니다.\n"
                    f"자세한 내용은 {dept} {name} {pos}님({phone})께 문의 부탁드립니다."
                )
            }

        return {
            "response": (
                "해당 질문에 대해 관련 문서와 담당자를 찾을 수 없습니다.\n"
            )
        }

    # 6️⃣ 문서 있음 → RAG + Gemini
    prompt = prompt_builder.run(documents=docs, question=question)["prompt"]
    answer = create_gemini_response(prompt)
    # 출처 정보 추가 
    # --- 🔥 출처 포맷팅 ---
    try:
        raw_name = docs[0].meta.get("file_name", "출처 정보 없음")
        # .pdf 제거
        if raw_name.lower().endswith(".pdf"):
            clean_name = raw_name[:-4]
        else:
            clean_name = raw_name

        answer += f"\n\n📄 출처: {clean_name}"

    except Exception:
        answer += "\n\n📄 출처: 알 수 없음"
    return {"response": answer}
    
    # except Exception as e:
    #     return {"response": f"서버 오류 발생: {str(e)}"}
    
@app.post("/api/faq")
async def faq(request: Request):
    data = await request.json()
    faq_number = data.get("faq_number")

    # 숫자가 유효한지 검사
    if faq_number is None or not isinstance(faq_number, int):
        return {"response": "FAQ 번호가 잘못 전달되었습니다."}

    # 리스트 범위 검사
    if faq_number < 0 or faq_number >= len(FIXED_FAQ_DATABASE):
        return {"response": "해당 FAQ 항목이 존재하지 않습니다."}

    # 해당 FAQ 답변을 반환
    return {"response": FIXED_FAQ_DATABASE[faq_number]}
