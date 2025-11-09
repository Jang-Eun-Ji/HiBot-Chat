import os
import pickle
import duckdb
import json
import numpy as np
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PyPDFToDocument
import google.generativeai as genai
from dotenv import load_dotenv

# --- 0. [필수] API 키 설정 ---
# .env 파일에서 환경변수 로드
load_dotenv()

# 환경변수에서 API 키 가져오기
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
else:
    print("⚠️  경고: GOOGLE_API_KEY가 설정되지 않았습니다.")
    # (API 키가 없어도 FAQ 기능은 작동합니다)

# --- 1. [신규] 규칙 기반 FAQ 데이터베이스 (Req 1 & 2) ---
# 기획안의 "Quick Reply" 및 "FAQ 자동 응답"용
# 키(Keyword)가 질문에 포함되어 있으면, AI(RAG)를 호출하지 않고 즉시 이 답변을 반환합니다.
# (키워드를 구체적으로 적을수록 좋습니다)
FIXED_FAQ_DATABASE = {
    "연차 어떻게 사용하나요?": "연차는... (미리 작성된 고정 답변)",
    "복무 규정 알려줘": "복무 규정은... (미리 작성된 고정 답변)",
    "경조사 휴가": "경조사 휴가 규정은 다음과 같습니다...",
    "출장 복명": "출장 복명은 그룹웨어의 '결재' 메뉴에서...",
    "법인카드 사용": "법인카드 사용 지침은..."
    # (여기에 5개의 Quick Reply 및 주요 FAQ 항목을 모두 추가하세요)
}
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
        text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)
        retriever = DuckDBEmbeddingRetriever(db_path=DB_PATH, top_k=5)
        print("✅ 임베더와 리트리버 초기화 완료")
    except Exception as e:
        print(f"❌ 임베더 초기화 실패: {e}")
        print("📋 해결방법:")
        print("   1. pip install --upgrade certifi")
        print("   2. 인터넷 연결 확인")
        return None
    
    prompt_template = """
    당신은 제공된 [문서] 내용을 바탕으로 답변하는 챗봇입니다.
    오직 [문서]에 있는 내용만을 근거로 사용자의 [질문]에 대해 대답해주세요.
    [문서]에 관련 내용이 없다면, "죄송합니다. 해당 문서에는 관련 내용이 없습니다."라고 정확하게 답변하세요.

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
    """Gemini API를 직접 사용하여 응답을 생성하는 함수 (기존 코드)"""
    try:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-pro')  # Updated model name
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

# --- 5. 챗봇 실행 ---
if __name__ == "__main__":
    # 챗봇 파이프라인 1회 초기화
    pipeline_components = initialize_chatbot()
    
    if pipeline_components:
        text_embedder, retriever, prompt_builder = pipeline_components
        
        # (테스트)
        
        # (1) FAQ 질문 (RAG 미사용)
        ask_chatbot("연차 어떻게 사용하나요?", text_embedder, retriever, prompt_builder)
        
        # (2) 문서 기반 질문 (RAG 사용)
        ask_chatbot("작년도 복무 규정 요약해줘.", text_embedder, retriever, prompt_builder)
        
        # (3) 문서에 없는 질문 (RAG 사용 -> 실패 응답)
        ask_chatbot("하늘은 왜 파란가요?", text_embedder, retriever, prompt_builder)