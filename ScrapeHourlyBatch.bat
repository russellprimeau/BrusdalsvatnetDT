@echo OFF
rem Run ScrapeHourly.py using the project's local .venv Python.
rem Using an absolute path to the interpreter avoids working-directory issues
rem when launched from Task Scheduler.

set SCRIPT_DIR=%~dp0
"%SCRIPT_DIR%.venv\Scripts\python.exe" "%SCRIPT_DIR%ScrapeHourly.py" >> "%SCRIPT_DIR%ScrapeHourly_stderr.log" 2>&1
