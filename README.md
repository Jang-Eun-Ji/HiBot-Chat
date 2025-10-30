**backend bash**

cd backend

python -m venv venv

window: source venv/Scripts/activate
mac: source venv/bin/activate

pip install -r requirements.txt

uvicorn main:app --reload --port 8000



백엔드 라이브러리 설치 후 

pip freeze > requirements.txt




**frontend bash**

cd frontend

npm install

npm start



인메모리
- 현재 백터 메모리로 
