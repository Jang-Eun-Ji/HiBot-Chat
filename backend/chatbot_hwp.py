# chatbot_hwp.py
# Updated chatbot to work with HWP document database

import os
import duckdb
import json
import numpy as np
from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack.dataclasses import Document
import google.generativeai as genai
from dotenv import load_dotenv

# --- 0. API 키 설정 ---
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if google_api_key:
    os.environ["GOOGLE_API_KEY"] = google_api_key
    genai.configure(api_key=google_api_key)
    print("✅ Google API 키가 설정되었습니다.")
else:
    print("⚠️ 경고: GOOGLE_API_KEY가 설정되지 않았습니다.")

# --- 1. 규칙 기반 FAQ 데이터베이스 ---
FIXED_FAQ_DATABASE = {
    "연차 어떻게 사용": "연차 사용은 그룹웨어 근태관리 시스템에서 신청하실 수 있습니다. 연차는 입사일 기준으로 매년 15일이 부여되며, 미사용 연차는 다음 해로 이월됩니다.",
    "출장 신청": "출장은 그룹웨어의 '결재' → '출장신청' 메뉴에서 신청하세요. 출장 완료 후 7일 이내에 출장보고서를 제출해야 합니다.",
    "법인카드 사용": "법인카드는 업무 관련 경비만 사용 가능하며, 사용 후 영수증과 함께 정산 처리해야 합니다.",
    "복무 규정": "출근시간은 오전 9시, 퇴근시간은 오후 6시이며, 점심시간은 12시~1시입니다. 지각 시 그룹웨어에서 지각사유를 입력해주세요.",
    "휴가 신청": "휴가는 그룹웨어 근태관리에서 사전 신청하시기 바랍니다. 경조사 휴가의 경우 관련 증빙서류가 필요합니다."
}

# --- 2. 경로 및 모델 설정 ---
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"
DB_PATH_SIMPLE = "hibot_store_simple.db"  # 간단 버전 (텍스트 검색)
DB_PATH_EMBEDDING = "hibot_store.db"      # 임베딩 버전

# --- 3. HWP 문서 검색기 ---
class HWPDocumentSearcher:
    """HWP 문서에서 텍스트 기반 검색을 수행"""
    
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = None
        self._connect()
    
    def _connect(self):
        try:
            self.conn = duckdb.connect(self.db_path)
            print(f"✅ 데이터베이스 연결 성공: {self.db_path}")
        except Exception as e:
            print(f"❌ 데이터베이스 연결 실패: {e}")
    
    def search_documents(self, query, limit=5):
        """텍스트 기반 문서 검색"""
        if not self.conn:
            return []
        
        try:
            # 간단한 단일 키워드 검색으로 변경
            sql = """
                SELECT id, content, meta,
                       length(content) - length(replace(lower(content), lower(?), '')) as relevance_score
                FROM documents 
                WHERE lower(content) LIKE lower(?)
                ORDER BY relevance_score DESC
                LIMIT ?
            """
            
            pattern = f"%{query}%"
            result = self.conn.execute(sql, (query, pattern, limit)).fetchall()
            
            documents = []
            for row in result:
                doc_id, content, meta_str, score = row
                meta = json.loads(meta_str) if meta_str else {}
                meta['search_score'] = score
                
                documents.append(Document(
                    id=doc_id,
                    content=content,
                    meta=meta
                ))
            
            return documents
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def get_statistics(self):
        """데이터베이스 통계 정보"""
        if not self.conn:
            return {}
        
        try:
            total_docs = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            
            file_stats = self.conn.execute("""
                SELECT JSON_EXTRACT_STRING(meta, '$.file_name') as filename, 
                       COUNT(*) as chunk_count
                FROM documents 
                GROUP BY filename 
                ORDER BY chunk_count DESC
                LIMIT 5
            """).fetchall()
            
            return {
                "total_documents": total_docs,
                "top_files": file_stats
            }
        except Exception as e:
            print(f"❌ 통계 조회 오류: {e}")
            return {}

# --- 4. Gemini API 응답 생성기 ---
class GeminiResponseGenerator:
    """Google Gemini API를 사용한 응답 생성"""
    
    def __init__(self):
        self.model_name = "gemini-1.5-flash"
        self.available = google_api_key is not None
    
    def generate_response(self, query, documents):
        """문서 컨텍스트를 바탕으로 응답 생성"""
        if not self.available:
            return "죄송합니다. AI 응답 생성 서비스가 현재 이용 불가능합니다. 시스템 관리자에게 문의하세요."
        
        try:
            # 컨텍스트 준비
            context_parts = []
            for i, doc in enumerate(documents[:3], 1):  # 상위 3개 문서만 사용
                filename = doc.meta.get('file_name', '알 수 없는 파일')
                content_preview = doc.content[:500]  # 처음 500자만
                context_parts.append(f"[문서 {i}: {filename}]\n{content_preview}")
            
            context = "\n\n".join(context_parts)
            
            # 프롬프트 구성
            prompt = f"""
다음은 조직 내부 규정 및 업무 관련 문서들입니다. 사용자의 질문에 대해 이 문서들을 참고하여 정확하고 도움이 되는 답변을 작성해주세요.

=== 관련 문서 내용 ===
{context}

=== 사용자 질문 ===
{query}

=== 답변 가이드라인 ===
1. 제공된 문서 내용을 바탕으로 정확한 정보를 제공하세요
2. 문서에서 찾을 수 없는 정보는 추측하지 마세요
3. 한국어로 친근하고 전문적인 톤으로 답변하세요
4. 필요시 해당 규정이나 지침의 제목을 언급하세요
5. 추가 문의가 필요한 경우 담당 부서 확인을 안내하세요

답변:
"""

            # Gemini 모델로 응답 생성
            model = genai.GenerativeModel(self.model_name)
            response = model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            print(f"❌ Gemini API 호출 실패: {e}")
            return f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

# --- 5. 통합 챗봇 클래스 ---
class HWPChatbot:
    """HWP 문서 기반 챗봇"""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or DB_PATH_SIMPLE
        self.searcher = HWPDocumentSearcher(self.db_path)
        self.generator = GeminiResponseGenerator()
        
        # 초기화 상태 확인
        stats = self.searcher.get_statistics()
        if stats:
            print(f"✅ 챗봇 초기화 완료 - 총 {stats.get('total_documents', 0)}개 문서 로드됨")
        else:
            print("⚠️ 문서 데이터베이스가 비어있거나 접근할 수 없습니다.")
    
    def check_fixed_faq(self, query):
        """고정 FAQ 확인"""
        query_lower = query.lower()
        for keyword, answer in FIXED_FAQ_DATABASE.items():
            if keyword.lower() in query_lower:
                return answer
        return None
    
    def chat(self, query):
        """메인 챗봇 응답 함수"""
        print(f"\n🤖 사용자 질문: {query}")
        
        # 1. 고정 FAQ 확인
        fixed_answer = self.check_fixed_faq(query)
        if fixed_answer:
            print("📋 고정 FAQ 응답 사용")
            return fixed_answer
        
        # 2. 문서 검색
        print("🔍 문서 검색 중...")
        documents = self.searcher.search_documents(query, limit=5)
        
        if not documents:
            return "죄송합니다. 관련 문서를 찾을 수 없습니다. 다른 키워드로 검색해보시거나 담당 부서에 직접 문의하시기 바랍니다."
        
        print(f"📄 {len(documents)}개 관련 문서 발견")
        
        # 3. AI 응답 생성
        print("🤖 AI 응답 생성 중...")
        response = self.generator.generate_response(query, documents)
        
        # 4. 참조 문서 정보 추가
        source_info = "\\n\\n**참조 문서:**\\n"
        for i, doc in enumerate(documents[:3], 1):
            filename = doc.meta.get('file_name', '알 수 없는 파일')
            source_info += f"{i}. {filename}\\n"
        
        return response + source_info
    
    def get_status(self):
        """챗봇 상태 정보"""
        stats = self.searcher.get_statistics()
        return {
            "database_connected": self.searcher.conn is not None,
            "ai_available": self.generator.available,
            "total_documents": stats.get('total_documents', 0),
            "top_files": stats.get('top_files', [])
        }

# --- 6. 테스트 함수 ---
def test_chatbot():
    """챗봇 테스트"""
    chatbot = HWPChatbot()
    
    print("=" * 50)
    print("🤖 HWP 챗봇 테스트 시작")
    print("=" * 50)
    
    # 상태 확인
    status = chatbot.get_status()
    print("📊 챗봇 상태:")
    print(f"  - 데이터베이스 연결: {status['database_connected']}")
    print(f"  - AI 사용 가능: {status['ai_available']}")
    print(f"  - 총 문서 수: {status['total_documents']}")
    print()
    
    # 테스트 질문들
    test_questions = [
        "연차 어떻게 사용하나요?",
        "휴직 관련 규정이 궁금해요",
        "출장 신청은 어떻게 하나요?",
        "급여 관련 문의가 있습니다",
        "인사평가는 언제 진행되나요?"
    ]
    
    for question in test_questions:
        print(f"❓ 질문: {question}")
        answer = chatbot.chat(question)
        print(f"🤖 답변: {answer[:200]}..." if len(answer) > 200 else f"🤖 답변: {answer}")
        print("-" * 40)

if __name__ == "__main__":
    test_chatbot()