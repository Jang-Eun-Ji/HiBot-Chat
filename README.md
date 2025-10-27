backend bash

cd backend
python -m venv venv
source venv/Scripts/activate
pip install fastapi uvicorn "fastapi-cors[standard]"
uvicorn main:app --reload --port 8000

frontend bash

cd frontend
npm install
npm start
