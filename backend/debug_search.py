# debug_search.py
import duckdb

conn = duckdb.connect('hibot_store_simple.db')

# 테스트 검색어들
test_queries = ['휴직', '급여', '인사평가', '연차']

for query in test_queries:
    result = conn.execute("SELECT COUNT(*) FROM documents WHERE lower(content) LIKE lower(?)", (f"%{query}%",)).fetchone()[0]
    print(f"'{query}' 관련 문서: {result}개")

# 샘플 검색 결과 확인
print("\n'휴직' 검색 결과 샘플:")
results = conn.execute("""
    SELECT JSON_EXTRACT_STRING(meta, '$.file_name') as filename, 
           SUBSTRING(content, 1, 100) as preview
    FROM documents 
    WHERE lower(content) LIKE lower('%휴직%')
    LIMIT 3
""").fetchall()

for filename, preview in results:
    print(f"- {filename}: {preview}...")

conn.close()