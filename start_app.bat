@echo off
echo Starting Titanic Survival Predictor...
cd /d "%~dp0"
start http://localhost:5000
python app.py
pause
