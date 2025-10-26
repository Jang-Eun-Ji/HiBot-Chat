from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware # CORS 미들웨어 추가
from pydantic import BaseModel

# FastAPI 앱 생성
app = FastAPI()

# CORS 설정 (중요!)
# 리액트 앱이 실행되는 http://localhost:3000 에서 오는 요청을 허용합니다.
origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,       # 허용할 출처
    allow_credentials=True,    # 쿠키 허용
    allow_methods=["*"],       # 모든 HTTP 메소드 허용
    allow_headers=["*"],       # 모든 HTTP 헤더 허용
)


# 1. 챗봇의 질문과 답변 데이터 (딕셔너리)
qa_database = {
    "안녕": "안녕하세요!",
    "이름이 뭐야?": "저는 FAQ 챗봇입니다.",
    "날씨 어때?": "저는 날씨는 잘 몰라요. 😅",
    "오늘 기분 어때": "저는 항상 좋습니다!",
    "프로젝트 주제": "파이썬과 리액트를 연동하는 것입니다."
}

# 2. 사용자가 보낼 요청 데이터 형식 정의
class ChatRequest(BaseModel):
    message: str


# 3. 챗봇 응답을 위한 POST 엔드포인트
@app.post("/api/chat")
async def handle_chat(request: ChatRequest):
    user_message = request.message
    
    # 4. 딕셔너리에서 답변 찾기
    # .get() 메소드를 사용하면, 키(질문)가 없을 경우 기본값(두 번째 인자)을 반환합니다.
    bot_response = qa_database.get(user_message, "죄송합니다. 무슨 말인지 잘 모르겠어요.")
    
    # 5. 찾은 답변을 JSON 형태로 리액트에 반환
    return {"response": bot_response}

# 리액트에서 호출할 API 엔드포인트
@app.get("/api/data")
def get_data():
    # 간단한 JSON 데이터 반환
    return {"message": "🎉 안녕하세요! 파이썬 백엔드에서 보낸 데이터입니다!"}