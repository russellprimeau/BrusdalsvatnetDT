@echo OFF
set SCRIPT_DIR=%~dp0
"%SCRIPT_DIR%venv\Scripts\python.exe" "%SCRIPT_DIR%ScrapeStep.py" >> "%SCRIPT_DIR%ScrapeStep_stderr.log" 2>&1
