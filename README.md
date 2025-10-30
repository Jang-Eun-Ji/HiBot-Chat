**backend bash**

cd backend

python -m venv venv

source venv/Scripts/activate

pip install -r requirements.txt

uvicorn main:app --reload --port 8000



백엔드 라이브러리 설치 후 

pip freeze > requirements.txt




**frontend bash**

cd frontend

npm install

npm start
