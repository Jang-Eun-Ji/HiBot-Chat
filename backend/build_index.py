# 색인 파일
import os
import argparse  # (1) 수동 실행 옵션을 받기 위해 추가
from haystack import Pipeline
# (2) DuckDB용 컴포넌트로 변경
from duckdb import DuckDBDocumentStore
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.converters import PyPDFToDocument

# --- 1. 경로 및 모델 설정 ---

# (3) ✨ 중요: 한국어 모델로 변경
# all-MiniLM-L6-v2 -> jhgan/ko-sbert-nli
EMBEDDING_MODEL = "jhgan/ko-sbert-nli" 
DB_PATH = "hibot_store.db" # (4) 영구 저장될 DB 파일 이름
DATA_PATH = "/Users/jang-eunji/Desktop/hibot-chat/hibot-chat-docs-pdf"

def main(force_rebuild=False):
    print("문서 색인을 시작합니다...")
    
    # --- 2. 영구 저장소(DuckDB) 초기화 ---
    document_store = DuckDBDocumentStore(db_path=DB_PATH)
    
    if force_rebuild:
        print(f"--force 옵션 감지. '{DB_PATH}'의 모든 문서를 삭제합니다.")
        document_store.delete_all_documents()

    # --- 3. 증분 색인 (Incremental Indexing) 로직 ---
    
    # (A) DB에 이미 저장된 파일 이름 목록 가져오기
    try:
        existing_docs = document_store.filter_documents()
        indexed_files = {doc.meta.get("file_name") for doc in existing_docs}
        print(f"현재 DB에 색인된 파일 수: {len(indexed_files)}")
    except Exception as e:
        print(f"DB 연결 오류 (처음 실행 시 정상): {e}")
        indexed_files = set()

    # (B) 실제 폴더에 있는 PDF 파일 목록 가져오기
    current_pdf_files = {f for f in os.listdir(DATA_PATH) if f.endswith(".pdf")}
    
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
    # (5) 한국어 모델로 임베더 초기화
    document_embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)
    document_embedder.warm_up() # 모델 로드

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