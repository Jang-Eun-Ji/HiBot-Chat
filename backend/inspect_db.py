# DB 검사 스크립트

import duckdb
import json

DB_PATH = "hibot_store.db"

conn = duckdb.connect(DB_PATH)

rows = conn.execute("""
    SELECT id, content, meta, embedding 
    FROM documents LIMIT 20
""").fetchall()



for row in rows:
    doc_id = row[0]
    content = row[1]
    meta_str = row[2]
    embedding = row[3]

    # META 보기 좋게 파싱
    try:
        meta = json.loads(meta_str) if meta_str else {}
        pretty_meta = json.dumps(meta, ensure_ascii=False, indent=2)
    except:
        pretty_meta = meta_str  # JSON 파싱 실패 시 원본 출력

    print("\nID:", doc_id)
    print("META:\n", pretty_meta)
    print("CONTENT:", content[:300], "...")
    print("EMBEDDING LEN:", len(embedding) if embedding else None)
    print("-" * 60)


# 1) 모든 문서 가져오기
# rows = conn.execute("SELECT meta FROM documents").fetchall()

# file_chunk_count = {}

# for (meta_str,) in rows:
#     try:
#         meta = json.loads(meta_str) if meta_str else {}
#     except:
#         meta = {}

#     file_name = meta.get("file_name", "UNKNOWN")

#     if file_name not in file_chunk_count:
#         file_chunk_count[file_name] = 0
#     file_chunk_count[file_name] += 1

# # ✅ 2) 출력
# print("\n===============================")
# print("📊 파일별 Chunk 개수")
# print("===============================\n")

# for file_name, chunk_count in sorted(file_chunk_count.items()):
#     print(f"📄 {file_name} → {chunk_count} chunks")

# print("\n===============================")
# print(f"✅ 총 파일 수: {len(file_chunk_count)}개")
# print(f"✅ 총 Chunk 수: {sum(file_chunk_count.values())}개")
# print("===============================\n")




conn.close()
