import os
import duckdb
import json
import numpy as np
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import Document
import google.generativeai as genai
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

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
    "[인사근태(확장)] → [시간외근무] → [시간외근무 신청관리] 메뉴에서 신청 가능합니다. 1일 최대 3시간 30분, 월 최대 15시간까지 신청 가능하며, 휴게시간 30분을 제외해 입력해야 합니다.",
    "급여 담당자 이메일로 가족수당 신청서와 증빙서류(가족관계증명서, 건강보험 자격확인서 등)를 제출하면 됩니다. 배우자 4만원, 직계존속·비속 각 3만원이 지급됩니다. ※ 관련근거: 보수규정 시행규칙 별표 제1호",
    "정규직·계약직 직원에게 연간 1,000,000포인트(1P=1원)가 부여되며, 단체보험료 공제 후 잔액 한도 내에서 사용 가능합니다. 입·퇴사자는 근무기간에 따라 월할 계산 적용됩니다.",
    "출장신청은 [인사근태(확장)] → [근태신청서] → [출장신청] 메뉴에서 가능합니다. 국내출장은 1주일 이내, 국외출장은 2주일 이내에 운임·숙박비 등 증빙서류를 첨부하여 정산해야 합니다.",
    "전산기기 및 사무기기(PC, 복합기, 세단기 등)는 기기 중간 또는 하단에 부착된 수리기사 연락처로 직접 유선 문의하시면 됩니다. 기타 시설물(조명, 의자, 문손잡이 등) 고장은 경영지원부 물품관리 담당자에게 연락해 주시기 바랍니다."
]

# --- 2. 경로 및 모델 설정 ---
# (2) ✨ 중요: build_index.py와 동일한 모델/저장소 경로 설정
# EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # build_index.py와 동일한 모델 사용
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"  # 한국어 모델 (SSL 문제 해결 후 사용)
DB_PATH = "hibot_store.db"  # build_index.py와 동일한 DuckDB 파일 경로

# --- 3. Custom DuckDB Retriever Class ---
class DuckDBEmbeddingRetriever:
    """DuckDB에서 유사한 문서를 검색하는 커스텀 리트리버"""
    
    def __init__(self, db_path, top_k=5):
        self.db_path = db_path
        self.top_k = top_k
        self.conn = None
        
    def connect(self):
        """DuckDB 연결"""
        if self.conn is None:
            self.conn = duckdb.connect(self.db_path)
    
    def run(self, query_embedding):
        """쿼리 임베딩과 유사한 문서들을 검색"""
        self.connect()
        
        # 모든 문서와 임베딩을 가져옴
        docs_data = self.conn.execute("""
            SELECT id, content, meta, embedding 
            FROM documents 
            WHERE embedding IS NOT NULL
        """).fetchall()
        
        if not docs_data:
            return {"documents": []}
        
        # 코사인 유사도 계산
        similarities = []
        for doc_id, content, meta_str, embedding in docs_data:
            if embedding:
                # 코사인 유사도 계산
                doc_embedding = np.array(embedding)
                query_emb = np.array(query_embedding[0])  # query_embedding is a list
                
                similarity = np.dot(query_emb, doc_embedding) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(doc_embedding)
                )
                
                try:
                    meta = json.loads(meta_str) if meta_str else {}
                except:
                    meta = {}
                
                similarities.append((similarity, doc_id, content, meta))
        
        # 유사도 순으로 정렬하고 top_k만 선택
        similarities.sort(reverse=True, key=lambda x: x[0])
        top_docs = similarities[:self.top_k]
        
        # Document 객체 생성
        documents = []
        for similarity, doc_id, content, meta in top_docs:
            doc = Document(id=doc_id, content=content, meta=meta)
            documents.append(doc)
        
        return {"documents": documents}
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
        # 질문의  임베딩과 코사인 유사도(similarity score)를 계산해서 가장 비슷한 문서 **5개(top_k=5)**를 반환함
        retriever = DuckDBEmbeddingRetriever(db_path=DB_PATH, top_k=5)
        print("✅ 임베더와 리트리버 초기화 완료")
    except Exception as e:
        print(f"❌ 임베더 초기화 실패: {e}")
        print("📋 해결방법:")
        print("   1. pip install --upgrade certifi")
        print("   2. 인터넷 연결 확인")
        return None
    
    prompt_template = """
    넌 제공된 [문서] 내용을 바탕으로 답변하는 챗봇이다.
    오직 [문서]에 있는 내용만을 근거로 [질문]에 대해 대답해.
    [문서]에 관련 내용이 없다면, "죄송합니다. 해당 문서에는 관련 내용이 없습니다."라고 정확하게 답변해.

    [문서]:
    {% for doc in documents %}
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
        model = genai.GenerativeModel('gemini-2.5-flash-lite')  # Updated to available model
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API 호출 중 오류 발생: {str(e)}"

def ask_chatbot(question, text_embedder, retriever, prompt_builder):
    """
    (✨ 신규 로직)
    사용자 질문을 받아서 FAQ(규칙)를 먼저 확인하고, 
    없으면 RAG 파이프라인을 실행하는 메인 "라우터"
    """
    print(f"\n[질문] 💬: {question}")
    
    # --- 1단계: 규칙 기반 FAQ 확인 (Req 1 & 2) ---
    # 기획안의 "키워드 포함 여부" 로직
    for keyword, answer in FIXED_FAQ_DATABASE.items():
        if keyword in question:
            print(f"[답변] 🤖 (규칙 기반 FAQ): {answer}")
            return answer

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
            print("[답변] 🤖 (RAG): 죄송합니다. 문서에서 관련 내용을 찾지 못했습니다.")
            return "죄송합니다. 문서에서 관련 내용을 찾지 못했습니다."

        # (C) 프롬프트 생성
        prompt_result = prompt_builder.run(documents=retrieved_docs, question=question)
        full_prompt = prompt_result["prompt"]
        
        # (D) Gemini API로 답변 생성
        answer = create_gemini_response(full_prompt)
        print(f"[답변] 🤖 (AI 생성): {answer}")
        return answer
        
    except Exception as e:
        error_msg = f"챗봇 실행 중 오류 발생: {str(e)}"
        print(f"[오류] ❌: {error_msg}")
        return error_msg

# --- 5. 백엔드 테스트용 챗봇 실행 ---
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
    for keyword, answer in FIXED_FAQ_DATABASE.items():
        if keyword in question:
            return {"response": answer}

    # 2️⃣ RAG + Gemini 호출
    try:
        query_emb = text_embedder.run(text=question)["embedding"]
        docs = retriever.run(query_embedding=[query_emb])["documents"]

        if not docs:
            return {"response": "죄송합니다. 문서에서 관련 내용을 찾지 못했습니다."}

        prompt = prompt_builder.run(documents=docs, question=question)["prompt"]
        answer = create_gemini_response(prompt)
        return {"response": answer}
    except Exception as e:
        return {"response": f"서버 오류 발생: {str(e)}"}
    
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
