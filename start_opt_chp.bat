@echo off
rem ---------------------------------------------------------------
rem opt_CHP - spusteni lokalniho weboveho UI (Streamlit).
rem Bez diakritiky zamerne: davka se cte pri ruznych kodovanich konzole.
rem Cestu k projektu si najde sama pres %~dp0, jde spustit dvojklikem.
rem ---------------------------------------------------------------
setlocal
cd /d "%~dp0"

if not defined OPT_CHP_DATA_DIR set "OPT_CHP_DATA_DIR=%LOCALAPPDATA%\py\opt_chp_data"
if not exist "%OPT_CHP_DATA_DIR%" mkdir "%OPT_CHP_DATA_DIR%"

set "LOG=%OPT_CHP_DATA_DIR%\run.log"
set "PY=.venv\Scripts\python.exe"
set "STAMP=.venv\install_ok.txt"

rem Interpret volame primo, nikdy pres activate - PowerShell ho blokuje ExecutionPolicy.
if exist "%STAMP%" goto run

echo [opt_chp] Prvni spusteni, pripravuji virtualni prostredi...
echo [opt_chp] Muze to trvat par minut, prubeh je v logu.
if not exist "%PY%" py -3.12 -m venv .venv
if errorlevel 1 goto nopython
if not exist "%PY%" goto nopython

echo [opt_chp] Instaluji balicky (pres proxy z HTTPS_PROXY)...
"%PY%" -m pip install -r requirements-lock.txt >>"%LOG%" 2>&1
if errorlevel 1 goto nopip
echo ok>"%STAMP%"

:run
echo [opt_chp] venv OK, spoustim Streamlit...
echo [opt_chp] log: %LOG%
echo.
"%PY%" -m streamlit run app.py
if errorlevel 1 goto norun
goto :eof

:nopython
echo.
echo [opt_chp] CHYBA: nepodarilo se vytvorit venv.
echo [opt_chp] Zkontroluj, ze funguje:  py -3.12 --version
pause
exit /b 1

:nopip
echo.
echo [opt_chp] CHYBA: instalace balicku selhala.
echo [opt_chp] Podrobnosti: %LOG%
echo [opt_chp] Casta pricina je nenastavena proxy - overit v NOVEM okne:
echo [opt_chp]   echo %%HTTPS_PROXY%%
pause
exit /b 1

:norun
echo.
echo [opt_chp] CHYBA: Streamlit skoncil s chybou.
echo [opt_chp] Podrobnosti: %LOG%
pause
exit /b 1
