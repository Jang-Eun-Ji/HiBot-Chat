# test_search_direct.py
from chatbot_hwp import HWPDocumentSearcher

# 직접 검색 테스트
searcher = HWPDocumentSearcher("hibot_store_simple.db")

test_queries = ["휴직", "급여", "인사"]

for query in test_queries:
    print(f"\n🔍 '{query}' 검색 테스트:")
    docs = searcher.search_documents(query, limit=3)
    print(f"결과: {len(docs)}개 문서")
    
    for i, doc in enumerate(docs, 1):
        filename = doc.meta.get('file_name', '알 수 없음')
        score = doc.meta.get('search_score', 0)
        preview = doc.content[:100].replace('\n', ' ')
        print(f"  {i}. {filename} (점수: {score})")
        print(f"     {preview}...")