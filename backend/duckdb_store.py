import duckdb, json, numpy as np
from haystack.dataclasses import Document

class DuckDBDocumentStore:
    def __init__(self, db_path: str):
        self.conn = duckdb.connect(db_path)
    def count_documents(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    def top_k_by_embedding(self, query_emb: np.ndarray, k: int = 5):
        # DuckDB는 리스트/배열을 DOUBLE[]로 저장했으므로 그대로 불러와 파이썬에서 점수 계산
        rows = self.conn.execute("SELECT id, content, meta, embedding FROM documents").fetchall()
        docs, scores = [], []
        for _id, content, meta_json, emb_list in rows:
            if emb_list is None: 
                continue
            emb = np.array(emb_list, dtype=float)
            # 코사인 유사도
            denom = (np.linalg.norm(emb) * np.linalg.norm(query_emb))
            sim = float(np.dot(emb, query_emb) / denom) if denom else -1.0
            try:
                meta = json.loads(meta_json) if meta_json else {}
            except:
                meta = {}
            docs.append(Document(id=_id, content=content, meta=meta, embedding=emb))
            scores.append(sim)
        if not docs:
            return []
        order = np.argsort(scores)[::-1][:k]
        return [docs[i] for i in order]

# Haystack 파이프라인 연결 없이 쓰기 위한 경량 리트리버 래퍼
class DuckDBEmbeddingRetriever:
    def __init__(self, document_store: DuckDBDocumentStore, top_k: int = 5):
        self.store = document_store
        self.top_k = top_k
    def run(self, query_embedding):
        docs = self.store.top_k_by_embedding(query_embedding, self.top_k)
        return {"documents": docs}
