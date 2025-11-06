# 색인 파일
import os
import argparse  # (1) 수동 실행 옵션을 받기 위해 추가
import duckdb
import json
from haystack import Pipeline
from haystack.dataclasses import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PyPDFToDocument

# --- 1. 경로 및 모델 설정 ---

# (3) ✨ 중요: 안정적인 모델 사용 (SSL 문제 해결 후 한국어 모델로 변경 가능)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 안정적인 영어 모델
# EMBEDDING_MODEL = "jhgan/ko-sbert-nli"  # 한국어 모델 (SSL 문제 해결 후 사용)
DB_PATH = "hibot_store.db"  # (4) Pure DuckDB 데이터베이스 파일
DATA_PATH = r"c:\Users\khis\Desktop\HiBot-Chat\hibot-chat-docs-pdf"  # Windows 경로

class DuckDBDocumentStore:
    """Pure DuckDB document storage implementation"""
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = duckdb.connect(db_path)
        self._setup_tables()
    
    def _setup_tables(self):
        """Create necessary tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id TEXT PRIMARY KEY,
                content TEXT,
                meta TEXT,
                embedding DOUBLE[]
            )
        """)
        self.conn.commit()
    
    def write_documents(self, documents):
        """Write documents directly to DuckDB"""
        for doc in documents:
            meta_json = json.dumps(doc.meta) if doc.meta else "{}"
            embedding_list = doc.embedding.tolist() if doc.embedding is not None else None
            
            self.conn.execute("""
                INSERT OR REPLACE INTO documents (id, content, meta, embedding)
                VALUES (?, ?, ?, ?)
            """, (str(doc.id), doc.content, meta_json, embedding_list))
        self.conn.commit()
        print(f"✅ {len(documents)}개 문서를 DuckDB에 저장했습니다.")
    
    def filter_documents(self, filters=None):
        """Get documents from DuckDB with optional filtering"""
        query = "SELECT id, content, meta, embedding FROM documents"
        
        # Add file_name filter if specified
        if filters and "meta" in filters:
            file_name = filters["meta"].get("file_name")
            if file_name:
                query += f" WHERE meta LIKE '%\"file_name\": \"{file_name}\"%'"
        
        result = self.conn.execute(query).fetchall()
        documents = []
        
        for row in result:
            doc_id, content, meta_str, embedding = row
            try:
                meta = json.loads(meta_str) if meta_str else {}
            except:
                meta = {}
                
            doc = Document(
                id=doc_id,
                content=content,
                meta=meta,
                embedding=embedding
            )
            documents.append(doc)
        
        return documents
    
    def count_documents(self):
        """Count documents in DuckDB"""
        result = self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()
        return result[0] if result else 0
    
    def delete_all_documents(self):
        """Delete all documents from DuckDB"""
        self.conn.execute("DELETE FROM documents")
        self.conn.commit()
        print("🗑️ 모든 문서가 DuckDB에서 삭제되었습니다.")

def main(force_rebuild=False):
    print("문서 색인을 시작합니다...")
    
    # --- 2. 영구 저장소(DuckDB) 초기화 ---
    try:
        document_store = DuckDBDocumentStore(db_path=DB_PATH)
        print(f"✅ DuckDB 저장소 '{DB_PATH}' 초기화 완료")
    except Exception as e:
        print(f"❌ DuckDB 저장소 초기화 실패: {e}")
        return
    
    if force_rebuild:
        print(f"--force 옵션 감지. '{DB_PATH}'의 모든 문서를 삭제합니다.")
        document_store.delete_all_documents()

    # --- 3. 증분 색인 (Incremental Indexing) 로직 ---
    
    # (A) DB에 이미 저장된 파일 이름 목록 가져오기
    try:
        existing_docs = document_store.filter_documents({})
        indexed_files = {doc.meta.get("file_name") for doc in existing_docs if doc.meta.get("file_name")}
        print(f"현재 DB에 색인된 파일 수: {len(indexed_files)}")
    except Exception as e:
        print(f"DB 연결 오류 (처음 실행 시 정상): {e}")
        indexed_files = set()

    # (B) 실제 폴더에 있는 PDF 파일 목록 가져오기
    if not os.path.exists(DATA_PATH):
        print(f"❌ 데이터 폴더를 찾을 수 없습니다: {DATA_PATH}")
        print("📋 해결방법: 다음 폴더를 생성하고 PDF 파일을 추가하세요:")
        print(f"   mkdir \"{DATA_PATH}\"")
        return
    
    try:
        current_pdf_files = {f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")}
    except Exception as e:
        print(f"❌ 폴더 읽기 오류: {e}")
        return
    
    # (C) 새로 추가된 파일만 필터링
    new_files_to_index = current_pdf_files - indexed_files
    
    if not new_files_to_index:
        print("✅ 새로 추가된 파일이 없습니다. 색인을 종료합니다.")
        return

    print(f"🚨 총 {len(new_files_to_index)}개의 새 파일을 찾았습니다. 색인을 진행합니다.")
    print(list(new_files_to_index))

    # --- 4. 색인 파이프라인 컴포넌트 준비 ---
    pdf_converter = PyPDFToDocument()
    splitter = DocumentSplitter(split_by="sentence", split_length=5)
    
    # (5) 임베더 초기화 (SSL 오류 처리 포함)
    try:
        document_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)
        document_embedder.warm_up()  # 모델 로드
        print(f"✅ 임베딩 모델 '{EMBEDDING_MODEL}' 로드 완료")
    except Exception as e:
        print(f"❌ 임베딩 모델 로드 실패: {e}")
        print("📋 해결방법:")
        print("   1. pip install --upgrade certifi")
        print("   2. 인터넷 연결 확인")
        print("   3. 기업 방화벽인 경우 IT 부서 문의")
        return

    # --- 5. 새 파일만 순회하며 색인 ---
    try:
        for file_name in new_files_to_index:
            print(f"처리 중: {file_name}...")
            file_path = os.path.join(DATA_PATH, file_name)
            
            # 1. PDF 변환
            docs = pdf_converter.run(sources=[file_path])["documents"]
            
            # 2. 메타데이터에 'file_name' 추가 (추적용)
            for doc in docs:
                doc.meta["file_name"] = file_name
            
            # 3. 문서 분할 (Chunking)
            split_docs = splitter.run(docs)["documents"]
            
            # 4. 임베딩 (로컬 실행)
            embedded_docs = document_embedder.run(split_docs)["documents"]
            
            # 5. DB에 저장 (영구)
            document_store.write_documents(embedded_docs)
            
        print(f"✅ {len(new_files_to_index)}개 파일의 색인 및 저장이 완료되었습니다.")
        print(f"📊 총 {document_store.count_documents()}개의 문서가 DuckDB에 저장되어 있습니다.")

    except Exception as e:
        print(f"❌ 문서 색인 중 오류 발생: {e}")

# --- 스크립트 실행 ---
if __name__ == "__main__":
    # (6) 수동으로 'python build_index.py --force' 실행 시 전체 재색인
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--force",
        action="store_true",
        help="DB를 강제로 비우고 모든 문서를 처음부터 다시 색인합니다."
    )
    args = parser.parse_args()
    
    main(force_rebuild=args.force)