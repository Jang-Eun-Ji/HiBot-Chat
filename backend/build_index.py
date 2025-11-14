# build_index.py
import os
import argparse
import json
import duckdb
import win32com.client
import tempfile

from haystack import Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder

from extract_text.extract_hwpx_text import extract_text_from_hwpx


# ------------------------------
# 1. 경로 설정
# ------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "hibot_store.db")
DATA_PATH = os.path.join(BASE_DIR, "../hibot-chat-docs-hwp")

EMBEDDING_MODEL = "jhgan/ko-sbert-nli"


# ------------------------------
# 2. DuckDB Document Store
# ------------------------------
class DuckDBDocumentStore:
    def __init__(self, db_path):
        self.conn = duckdb.connect(db_path)
        self._setup_tables()

    def _setup_tables(self):
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
        for doc in documents:
            meta_json = json.dumps(doc.meta) if doc.meta else "{}"

            # embedding → Python list로 처리
            if doc.embedding is not None:
                if hasattr(doc.embedding, "tolist"):
                    embed_list = doc.embedding.tolist()
                else:
                    embed_list = list(doc.embedding)
            else:
                embed_list = None

            self.conn.execute("""
                INSERT OR REPLACE INTO documents (id, content, meta, embedding)
                VALUES (?, ?, ?, ?)
            """, (str(doc.id), doc.content, meta_json, embed_list))

        self.conn.commit()
        print(f"✅ {len(documents)}개 문서를 DB에 저장했습니다.")

    def filter_documents(self, filters=None):
        query = "SELECT id, content, meta, embedding FROM documents"
        result = self.conn.execute(query).fetchall()

        documents = []
        for row in result:
            doc_id, content, meta_str, embedding = row
            meta = json.loads(meta_str) if meta_str else {}

            documents.append(Document(
                id=doc_id,
                content=content,
                meta=meta,
                embedding=embedding
            ))
        return documents

    def delete_all_documents(self):
        self.conn.execute("DELETE FROM documents")
        self.conn.commit()
        print("🗑️ 모든 문서를 삭제했습니다.")

    def count_documents(self):
        return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


# ------------------------------
# 3. HWP → Text 변환기
# ------------------------------
def convert_hwp_to_hwpx(hwp_path):
    """HWP 파일을 임시 HWPX 파일로 변환"""
    try:
        hwp = win32com.client.Dispatch("HWPFrame.HwpObject")
        
        # 임시 HWPX 파일 경로 생성
        temp_dir = tempfile.gettempdir()
        hwpx_filename = os.path.splitext(os.path.basename(hwp_path))[0] + ".hwpx"
        hwpx_path = os.path.join(temp_dir, hwpx_filename)
        
        # HWP 파일 열기
        hwp.Open(hwp_path, "HWP", "forceopen:true")
        
        # HWPX 형식으로 저장
        hwp.SaveAs(hwpx_path, "HWPX", "version:1.0")
        
        # 문서 닫기
        hwp.XHwpDocuments.Close(isDirty=False)
        hwp.Quit()
        
        return hwpx_path
    except Exception as e:
        print(f"❌ HWP → HWPX 변환 실패 ({hwp_path}): {e}")
        return None

def extract_text_from_hwp(hwp_path):
    """HWP 파일에서 텍스트 추출"""
    # HWP → HWPX 변환
    hwpx_path = convert_hwp_to_hwpx(hwp_path)
    if not hwpx_path:
        return ""
    
    try:
        # HWPX에서 텍스트 추출
        text = extract_text_from_hwpx(hwpx_path)
        
        # 임시 파일 정리
        if os.path.exists(hwpx_path):
            os.remove(hwpx_path)
        
        return text
    except Exception as e:
        print(f"❌ HWPX 텍스트 추출 실패 ({hwpx_path}): {e}")
        # 임시 파일 정리
        if os.path.exists(hwpx_path):
            os.remove(hwpx_path)
        return ""


# ------------------------------
# 4. HWP → Haystack Document 변환
# ------------------------------
def convert_hwp_to_documents(hwp_path, file_name):
    text = extract_text_from_hwp(hwp_path)
    return [
        Document(
            content=text,
            meta={"file_name": file_name}
        )
    ]


# ------------------------------
# 5. 메인 색인 로직
# ------------------------------
def main(force_rebuild=False):
    print("DATA_PATH:", DATA_PATH)
    print("문서 색인을 시작합니다...")

    # 1) DB 초기화
    store = DuckDBDocumentStore(DB_PATH)

    # --force 옵션이면 전체 삭제
    if force_rebuild:
        print("⚠️ --force 옵션 감지 → 전체 문서 삭제 중…")
        store.delete_all_documents()

    # DB에 저장된 파일 목록
    existing_docs = store.filter_documents()
    indexed_files = {d.meta.get("file_name") for d in existing_docs if d.meta.get("file_name")}

    print(f"✅ DB에 기록된 HWP 파일 수: {len(indexed_files)}")

    # 실제 폴더에 존재하는 HWP 목록
    if not os.path.exists(DATA_PATH):
        print("❌ HWP 폴더가 없습니다:", DATA_PATH)
        return

    hwp_files = {f for f in os.listdir(DATA_PATH) if f.lower().endswith(".hwp")}
    new_files = hwp_files - indexed_files

    if not new_files:
        print("✅ 새로 색인할 HWP 파일이 없습니다.")
        return

    print(f"🚨 새 HWP 발견 → {len(new_files)}개 색인 시작: {list(new_files)}")

    # 문서 분할기
    splitter = DocumentSplitter(split_by="sentence", split_length=5)
    splitter.warm_up()

    # 문서 임베딩 모델
    embedder = SentenceTransformersDocumentEmbedder(model=EMBEDDING_MODEL)
    embedder.warm_up()

    # ✅ 새 파일들 색인
    for file_name in new_files:
        print(f"📄 처리 중: {file_name}")

        hwp_path = os.path.join(DATA_PATH, file_name)

        # (1) HWP → Document 변환
        docs = convert_hwp_to_documents(hwp_path, file_name)

        # (2) 문장 단위 chunking
        split_docs = splitter.run(docs)["documents"]

        # (3) 임베딩
        embedded_docs = embedder.run(split_docs)["documents"]

        # (4) DB 저장
        store.write_documents(embedded_docs)

    print("✅ 모든 새 HWP 색인이 완료되었습니다.")
    print("📊 총 문서 수:", store.count_documents())


# ------------------------------
# 6. 실행부
# ------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="모든 문서를 삭제 후 전체 재색인")
    args = parser.parse_args()

    main(force_rebuild=args.force)
