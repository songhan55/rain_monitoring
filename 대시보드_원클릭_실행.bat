@echo off
chcp 65001 > nul
title AI 도로 위험 감지 관제 대시보드 실행기
echo =======================================================
echo   🌧️ Rain Monitoring: 도로 위험 감지 AI 관제 대시보드
echo =======================================================
echo.
echo [1/2] 서버 상태를 확인하고 있습니다...

:: 5000번 포트 확인
netstat -ano | findstr :5000 > nul
if %errorlevel% equ 0 (
    echo [*] 대시보드 서버가 이미 백그라운드에서 실행 중입니다.
) else (
    echo [*] 대시보드 서버를 시작합니다...
    start /min python dashboard_app.py
    timeout /t 3 /nobreak > nul
)

echo [2/2] 웹 브라우저에서 대시보드를 엽니다...
start http://127.0.0.1:5000/

echo.
echo ✅ 대시보드가 성공적으로 열렸습니다! (창을 닫으셔도 됩니다)
timeout /t 3 > nul
exit
