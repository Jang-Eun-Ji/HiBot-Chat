import os
from haystack import Pipeline
# from haystack.components.generators import OpenAIGenerator  # OpenAI 
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder  # Using sentence transformers for embedding
from haystack.components.builders import PromptBuilder
from duckdb import DuckDBDocumentStore
from duckdb import DuckDBEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder
# from haystack.document_stores.in_memory import InMemoryDocumentStore
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
# (2) ✨ 중요: build_index.py와 동일한 모델/DB 경로 설정
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"
DB_PATH = "hibot_store.db"
# --- 3. [신규] RAG 파이프라인 "라우터" (Req 3) ---

def initialize_chatbot():
    print("챗봇 초기화 중...")
    
    # (A) 영구 저장소(DuckDB) 연결 (읽기 전용)
    try:
        document_store = DuckDBDocumentStore(db_path=DB_PATH)
        print(f"✅ '{DB_PATH}'에서 {document_store.count_documents()}개 문서를 불러왔습니다.")
    except Exception as e:
        print(f"❌ '{DB_PATH}' DB 파일을 찾을 수 없습니다. {e}")
        print("먼저 'python build_index.py' 스크립트를 실행하여 문서를 색인해주세요.")
        return None

    # (B) RAG 파이프라인 준비 (기존 코드와 유사)
    text_embedder = SentenceTransformersTextEmbedder(model=EMBEDDING_MODEL)
    retriever = DuckDBEmbeddingRetriever(document_store=document_store, top_k=5)
    
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
    prompt_builder = PromptBuilder(template=prompt_template)
    
    # (C) 검색 전용 파이프라인 구축 (생성기는 별도 처리)
    search_pipeline = Pipeline()
    search_pipeline.add_component("query_embedder", text_embedder)
    search_pipeline.add_component("retriever", retriever)
    search_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")
    text_embedder.warm_up()
    
    print("✅ 챗봇 RAG 파이프라인 준비 완료.")
    return search_pipeline, prompt_builder

def create_gemini_response(prompt):
    """Gemini API를 직접 사용하여 응답을 생성하는 함수 (기존 코드)"""
    try:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API 호출 중 오류 발생: {str(e)}"

def ask_chatbot(question, search_pipeline, prompt_builder):
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
        # (A) 관련 문서 검색
        search_result = search_pipeline.run({"query_embedder": {"text": question}})
        retrieved_docs = search_result["retriever"]["documents"]
        
        if not retrieved_docs:
            print("[답변] 🤖 (RAG): 죄송합니다. 문서에서 관련 내용을 찾지 못했습니다.")
            return "죄송합니다. 문서에서 관련 내용을 찾지 못했습니다."

        # (B) 프롬프트 생성
        prompt_result = prompt_builder.run(documents=retrieved_docs, question=question)
        full_prompt = prompt_result["prompt"]
        
        # (C) Gemini API로 답변 생성
        answer = create_gemini_response(full_prompt)
        print(f"[답변] 🤖 (AI 생성): {answer}")
        return answer
        
    except Exception as e:
        error_msg = f"챗봇 실행 중 오류 발생: {str(e)}"
        print(f"[오류] ❌: {error_msg}")
        return error_msg

# --- 4. 챗봇 실행 ---
if __name__ == "__main__":
    # 챗봇 파이프라인 1회 초기화
    pipeline_components = initialize_chatbot()
    
    if pipeline_components:
        search_pipeline, prompt_builder = pipeline_components
        
        # (테스트)
        
        # (1) FAQ 질문 (RAG 미사용)
        ask_chatbot("연차 어떻게 사용하나요?", search_pipeline, prompt_builder)
        
        # (2) 문서 기반 질문 (RAG 사용)
        ask_chatbot("작년도 복무 규정 요약해줘.", search_pipeline, prompt_builder)
        
        # (3) 문서에 없는 질문 (RAG 사용 -> 실패 응답)
        ask_chatbot("하늘은 왜 파란가요?", search_pipeline, prompt_builder)