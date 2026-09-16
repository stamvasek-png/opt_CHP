@echo off
rem ---------------------------------------------------------------
rem opt_CHP - aktualizace z rozbaleneho ZIPu.
rem
rem POZOR: tento skript musi lezet MIMO aktualizovanou slozku, jinak by
rem se prepsal sam za behu a cmd by se na tom rozsypal. Zkopiruj ho do
rem %LOCALAPPDATA%\py\ a poustej odtamtud.
rem
rem Pouziti:  update_opt_chp.bat "cesta\k\rozbalene\slozce"
rem ---------------------------------------------------------------
setlocal
set "SRC=%~1"
if not defined OPT_CHP_HOME set "OPT_CHP_HOME=%LOCALAPPDATA%\py\opt_chp"
set "STAMP=%OPT_CHP_HOME%\.venv\install_ok.txt"
set "OLDLOCK=%TEMP%\opt_chp_lock_old.txt"

if "%SRC%"=="" goto usage
if not exist "%SRC%\app.py" goto badsrc
if not exist "%OPT_CHP_HOME%\app.py" goto baddest

echo [opt_chp] Zdroj: %SRC%
echo [opt_chp] Cil:   %OPT_CHP_HOME%
echo.

rem Stary lockfile si odlozime, at pozname zmenu zavislosti.
copy /y "%OPT_CHP_HOME%\requirements-lock.txt" "%OLDLOCK%" >nul 2>&1

rem /MIR srovna cil se zdrojem; .venv a __pycache__ zustanou nedotcene.
rem Cache uz ve slozce s kodem neni (OPT_CHP_DATA_DIR), nema se co ztratit.
set "SKIP=/XD .venv .git __pycache__ /XF *.pyc"
robocopy "%SRC%" "%OPT_CHP_HOME%" /MIR %SKIP% /NFL /NDL /NJH /NJS
if errorlevel 8 goto failed

rem Zmenil-li se lockfile, zrusime znamku - balicky se pri startu doinstaluji.
fc "%OLDLOCK%" "%OPT_CHP_HOME%\requirements-lock.txt" >nul 2>&1
if errorlevel 1 goto relock
echo.
echo [opt_chp] Hotovo, zavislosti beze zmeny.
goto :eof

:relock
if exist "%STAMP%" del "%STAMP%" >nul 2>&1
echo.
echo [opt_chp] Hotovo. Zavislosti se zmenily - pri pristim
echo [opt_chp] spusteni se automaticky doinstaluji.
goto :eof

:usage
echo Pouziti: update_opt_chp.bat "cesta\k\rozbalene\slozce"
exit /b 1

:badsrc
echo [opt_chp] CHYBA: v "%SRC%" neni app.py - to neni slozka s projektem.
exit /b 1

:baddest
echo [opt_chp] CHYBA: v "%OPT_CHP_HOME%" neni app.py.
echo [opt_chp] Nastav OPT_CHP_HOME na spravnou slozku.
exit /b 1

:failed
echo [opt_chp] CHYBA: robocopy selhal.
exit /b 1
