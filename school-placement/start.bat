@echo off
cd /d "C:\Users\Мария\Desktop\ИИ 3 семестр\python-package-boilerplate\school-placement"
call .venv\Scripts\activate.bat
streamlit run app.py --server.headless true --server.port 8501
