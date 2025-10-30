import os
from haystack import Pipeline
from haystack.components.generators import OpenAIGenerator  # Using OpenAI instead of Gemini for now
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder  # Using sentence transformers for embedding
from haystack.components.builders import PromptBuilder
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PyPDFToDocument
import google.generativeai as genai
from dotenv import load_dotenv

# --- 0. [필수] API 키 설정 ---
# .env 파일에서 환경변수 로드
load_dotenv()

# 환경변수에서 API 키 가져오기
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key

# (만약 키가 설정되지 않았다면 경고 메시지만 표시)
if not os.environ.get("GOOGLE_API_KEY") or os.environ.get("GOOGLE_API_KEY") == "YOUR_GOOGLE_API_KEY_HERE":
    print("⚠️  경고: GOOGLE_API_KEY가 설정되지 않았습니다.")
    print("   Gemini API 기능을 사용하려면 실제 API 키를 설정해주세요.")
    print("   현재는 검색 기능만 테스트할 수 있습니다.")


# --- 1. 핵심 컴포넌트 초기화 ---

# (A) 벡터 저장소 (문서와 벡터를 저장할 창고)
#     (간단한 테스트를 위해 '메모리' 기반 저장소 사용)
document_store = InMemoryDocumentStore()

# (B) 텍스트 임베더 (무료 SentenceTransformers 임베딩 모델 사용)
#     로컬에서 실행되므로 데이터가 외부로 전송되지 않습니다.
text_embedder = SentenceTransformersTextEmbedder(model="all-MiniLM-L6-v2")
document_embedder = SentenceTransformersDocumentEmbedder(model="all-MiniLM-L6-v2")

# (C) 리트리버 (검색기)
retriever = InMemoryEmbeddingRetriever(document_store=document_store, top_k=5) # 5개 조각 검색

# (D) 프롬프트 빌더 (Gemini에게 보낼 질문지)
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

# (E) 생성기 (OpenAI GPT - 답변 생성용, 또는 직접 Gemini API 사용)
# 참고: OpenAI API 키가 필요합니다. 환경변수 OPENAI_API_KEY 설정 필요
# generator = OpenAIGenerator(model="gpt-3.5-turbo")

# 대신 직접 Gemini API를 사용하는 함수를 만들겠습니다
def create_gemini_response(prompt):
    """Gemini API를 직접 사용하여 응답을 생성하는 함수"""
    try:
        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini API 호출 중 오류 발생: {str(e)}"


# --- 2. [파일 저장] 문서 색인(Indexing) 파이프라인 ---
# 1단계에서 PDF로 변환한 파일들이 있는 경로 (Mac 경로로 수정)
data_path = "/Users/jang-eunji/Desktop/hibot-chat/hibot-chat-docs-pdf"  # 실제 데이터 폴더 경로로 수정하세요

# 해당 경로에서 .pdf 파일만 리스트업
try:
    pdf_files = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"경고: '{data_path}'에서 PDF 파일을 찾을 수 없습니다.")
        print("1단계: HWP를 PDF로 변환하는 작업을 완료했는지 확인하세요.")
    else:
        print(f"총 {len(pdf_files)}개의 PDF 파일을 찾았습니다. 색인을 시작합니다...")
        
        # 1. PDF -> Document 객체로 변환
        pdf_converter = PyPDFToDocument()
        all_docs = pdf_converter.run(sources=pdf_files)["documents"]
        
        # 2. 긴 문서를 작은 조각(chunk)으로 자르기 (RAG의 핵심)
        #    (여기서는 5개의 문장(sentence) 단위로 자름)
        splitter = DocumentSplitter(split_by="sentence", split_length=5)
        splitter.warm_up()  # 컴포넌트 초기화
        split_docs = splitter.run(all_docs)["documents"]

        # 3. SentenceTransformers 임베더로 임베딩 실행 (로컬에서 처리)
        #    (이때 문서 조각들이 로컬에서 처리되므로 외부로 전송되지 않습니다)
        document_embedder.warm_up()  # 임베더 초기화
        embedded_docs = document_embedder.run(split_docs)["documents"]

        # 4. 저장소에 쓰기
        document_store.write_documents(embedded_docs)
        
        print(f"✅ {len(embedded_docs)}개의 문서 조각을 성공적으로 색인했습니다.")

except Exception as e:
    print(f"❌ 문서 색인 중 오류 발생: {e}")

# --- 3. [질문 처리] RAG 파이프라인 구축 ---

# 검색 전용 파이프라인 구축 (생성기는 별도 처리)
search_pipeline = Pipeline()

# 파이프라인에 컴포넌트 추가
search_pipeline.add_component("query_embedder", text_embedder) # 1. 질문 임베딩
search_pipeline.add_component("retriever", retriever)       # 2. 문서 검색

# 컴포넌트 연결
search_pipeline.connect("query_embedder.embedding", "retriever.query_embedding")

# 검색용 임베더 초기화
text_embedder.warm_up()


# --- 4. 챗봇 실행 함수 ---

def ask_chatbot(question):
    print(f"\n[질문] 💬: {question}")
    
    try:
        # 1단계: 관련 문서 검색
        search_result = search_pipeline.run({"query_embedder": {"text": question}})
        retrieved_docs = search_result["retriever"]["documents"]
        
        # 2단계: 프롬프트 생성
        prompt_result = prompt_builder.run(documents=retrieved_docs, question=question)
        full_prompt = prompt_result["prompt"]
        
        # 3단계: Gemini API로 답변 생성
        answer = create_gemini_response(full_prompt)
        print(f"[답변] 🤖: {answer}")
        return answer
        
    except Exception as e:
        error_msg = f"챗봇 실행 중 오류 발생: {str(e)}"
        print(f"[오류] ❌: {error_msg}")
        return error_msg

# --- [테스트] 챗봇에 질문하기 ---
if document_store.count_documents() > 0:
    ask_chatbot("문서에 있는 내용으로 질문해보세요.")
    ask_chatbot("또 다른 질문을 해보세요.")
    ask_chatbot("하늘은 왜 파란가요?") # (문서에 없을 내용)
else:
    print("\n[알림] 문서가 색인되지 않아 챗봇을 실행할 수 없습니다. 스크립트 상단의 PDF 경로와 API 키를 확인하세요.")