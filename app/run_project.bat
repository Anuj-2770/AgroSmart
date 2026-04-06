@echo off
cd /d C:\Users\anuj_\OneDrive\Desktop\Agriculture-Optimization-System-main
call venv37\Scripts\activate
start "" cmd /c "python app\app.py"
timeout /t 3 >nul
start http://127.0.0.1:5000