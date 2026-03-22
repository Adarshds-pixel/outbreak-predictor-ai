@echo off
echo Starting Backend...

:: Go to backend folder
cd backend

:: Open browser automatically
start "" http://127.0.0.1:8000/

:: Run FastAPI using Python (because uvicorn is inside app.py)
python app.py

pause
