@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto directory_error

rem Keep this file ASCII with Windows CRLF line endings.
rem Chinese text and style defaults are defined in main.py.
if exist ".venv\Scripts\python.exe" goto local_python
where py >nul 2>nul
if not errorlevel 1 goto python_launcher
where python >nul 2>nul
if not errorlevel 1 goto system_python
echo Python was not found. Install Python 3.10+ and run install.bat.
set "launch_error=1"
goto failed

:local_python
".venv\Scripts\python.exe" -u main.py --gui %*
goto check_result

:python_launcher
py -3 -u main.py --gui %*
goto check_result

:system_python
python -u main.py --gui %*

:check_result
set "launch_error=%errorlevel%"
if "%launch_error%"=="0" exit /b 0
echo.
echo Fourier Studio could not start. See the error above.
echo For missing dependencies, run install.bat and try again.
goto failed

:directory_error
echo Could not open the project directory.
set "launch_error=1"

:failed
echo.
pause
exit /b %launch_error%
