# inspect_hwp_db.py
# Script to inspect the HWP database contents

import duckdb
import json
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "hibot_store_simple.db")

def inspect_db():
    print("🔍 HWP 데이터베이스 검사")
    print("=" * 50)
    
    conn = duckdb.connect(DB_PATH)
    
    # 총 문서 수
    total_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    print(f"📊 총 문서 청크 수: {total_docs}")
    
    # 파일별 통계
    print("\n📁 파일별 청크 수:")
    file_stats = conn.execute("""
        SELECT JSON_EXTRACT_STRING(meta, '$.file_name') as filename, 
               COUNT(*) as chunk_count,
               AVG(LENGTH(content)) as avg_length
        FROM documents 
        GROUP BY filename 
        ORDER BY chunk_count DESC
        LIMIT 10
    """).fetchall()
    
    for filename, count, avg_len in file_stats:
        print(f"  {filename}: {count}개 청크 (평균 {avg_len:.0f}자)")
    
    # 검색 테스트
    print("\n🔍 검색 테스트:")
    search_terms = ["인사", "급여", "휴가", "출장", "계약"]
    
    for term in search_terms:
        results = conn.execute("""
            SELECT COUNT(*) 
            FROM documents 
            WHERE lower(content) LIKE lower(?)
        """, (f"%{term}%",)).fetchone()[0]
        print(f"  '{term}' 검색 결과: {results}개 문서")
    
    # 샘플 문서 내용
    print("\n📄 샘플 문서 내용:")
    sample = conn.execute("""
        SELECT JSON_EXTRACT_STRING(meta, '$.file_name') as filename,
               SUBSTRING(content, 1, 200) as sample_content
        FROM documents 
        LIMIT 3
    """).fetchall()
    
    for i, (filename, content) in enumerate(sample, 1):
        print(f"  {i}. {filename}")
        print(f"     {content}...")
    
    conn.close()

def search_documents(query, limit=5):
    """문서 검색 테스트"""
    print(f"\n🔍 '{query}' 검색 결과:")
    print("=" * 50)
    
    conn = duckdb.connect(DB_PATH)
    
    sql = """
        SELECT JSON_EXTRACT_STRING(meta, '$.file_name') as filename,
               content,
               length(content) - length(replace(lower(content), lower(?), '')) as relevance
        FROM documents 
        WHERE lower(content) LIKE lower(?)
        ORDER BY relevance DESC
        LIMIT ?
    """
    
    pattern = f"%{query}%"
    results = conn.execute(sql, (query, pattern, limit)).fetchall()
    
    if results:
        for i, (filename, content, score) in enumerate(results, 1):
            print(f"{i}. 📄 {filename} (관련도: {score})")
            # 검색어 주변 텍스트 하이라이트
            content_lower = content.lower()
            query_lower = query.lower()
            pos = content_lower.find(query_lower)
            if pos != -1:
                start = max(0, pos - 50)
                end = min(len(content), pos + len(query) + 50)
                snippet = content[start:end].replace('\n', ' ')
                print(f"   💡 {snippet}")
            print()
    else:
        print("❌ 검색 결과가 없습니다.")
    
    conn.close()

if __name__ == "__main__":
    if os.path.exists(DB_PATH):
        inspect_db()
        
        # 대화형 검색 테스트
        while True:
            query = input("\n🔍 검색할 단어를 입력하세요 (종료: quit): ").strip()
            if query.lower() in ['quit', 'exit', '종료', 'q']:
                break
            if query:
                search_documents(query)
    else:
        print(f"❌ 데이터베이스 파일이 없습니다: {DB_PATH}")
        print("먼저 build_index_simple.py를 실행하세요.")