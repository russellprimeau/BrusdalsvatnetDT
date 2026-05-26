@echo OFF
set SCRIPT_DIR=%~dp0
"%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%ScrapeWeather.py" >> "%SCRIPT_DIR%ScrapeWeather_stderr.log" 2>&1
