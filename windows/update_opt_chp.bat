@echo off
rem ---------------------------------------------------------------
rem opt_CHP - aktualizace ze stazeneho ZIPu.
rem
rem Staci dvojklik: skript si najde nejnovejsi opt_CHP-*.zip ve slozce
rem Stazene soubory, rozbali ho a po potvrzeni prepise slozku s kodem.
rem
rem Volitelne jde predat cestu k uz rozbalene slozce:
rem   update_opt_chp.bat "cesta\k\rozbalene\slozce"
rem
rem POZOR: skript musi lezet MIMO aktualizovanou slozku, jinak by se
rem prepsal sam za behu a cmd by se na tom rozsypal. Patri do
rem %LOCALAPPDATA%\py\ - kopie z repozitare skonci v opt_chp\windows\,
rem coz je jiny soubor, takze si nelezou do cesty.
rem ---------------------------------------------------------------
setlocal
if not defined OPT_CHP_HOME set "OPT_CHP_HOME=%LOCALAPPDATA%\py\opt_chp"
set "DL=%USERPROFILE%\Downloads"
set "WORK=%LOCALAPPDATA%\py\_opt_chp_update"
set "STAMP=%OPT_CHP_HOME%\.venv\install_ok.txt"
set "OLDLOCK=%TEMP%\opt_chp_lock_old.txt"
set "SRC=%~1"
set "ZIP="

if not exist "%OPT_CHP_HOME%\app.py" goto baddest
if not "%SRC%"=="" goto havesrc

rem --- Najdi nejnovejsi ZIP (dir /o-d radi od nejnovejsiho) ---
for /f "delims=" %%F in ('dir /b /o-d "%DL%\opt_CHP-*.zip" 2^>nul') do (
    set "ZIP=%DL%\%%F"
    goto gotzip
)
goto nozip

:gotzip
echo [opt_chp] ZIP: %ZIP%
echo [opt_chp] Cil: %OPT_CHP_HOME%
echo.
set "ANS="
set /p ANS=Prepsat slozku s kodem? [A/N] 
rem Bereme A i Y - na ceske klavesnici clovek stejne casto napise "y".
rem Prazdna odpoved (jen Enter) znamena ne, at je vychozi stav bezpecny.
if /i "%ANS%"=="A" goto confirmed
if /i "%ANS%"=="ANO" goto confirmed
if /i "%ANS%"=="Y" goto confirmed
if /i "%ANS%"=="YES" goto confirmed
goto cancelled

:confirmed

rem --- Rozbal do docasne slozky ---
if exist "%WORK%" rd /s /q "%WORK%"
mkdir "%WORK%"
echo.
echo [opt_chp] Rozbaluji...
tar -xf "%ZIP%" -C "%WORK%"
if errorlevel 1 goto untarfail

rem V ZIPu je jedna slozka, jmenuje se podle vetve (opt_CHP-main apod.)
for /d %%D in ("%WORK%\*") do set "SRC=%%D"
if not exist "%SRC%\app.py" goto badzip

:havesrc
if not exist "%SRC%\app.py" goto badsrc
echo [opt_chp] Zdroj: %SRC%
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

call :cleanup
echo.
echo [opt_chp] Hotovo. Zavislosti se nezmenily.
pause
exit /b 0

:relock
if exist "%STAMP%" del "%STAMP%" >nul 2>&1
call :cleanup
echo.
echo [opt_chp] Hotovo. Zavislosti se zmenily - pri pristim spusteni
echo [opt_chp] aplikace se automaticky doinstaluji.
pause
exit /b 0

:cleanup
if exist "%WORK%" rd /s /q "%WORK%" >nul 2>&1
goto :eof

:nozip
echo [opt_chp] CHYBA: ve slozce Stazene soubory neni zadny opt_CHP-*.zip
echo [opt_chp] Hledal jsem v: %DL%
echo.
echo [opt_chp] Stahni ZIP na GitHubu pres Code - Download ZIP a spust znovu.
pause
exit /b 1

:cancelled
echo.
echo [opt_chp] Zruseno - slozka s kodem zustala nedotcena.
echo [opt_chp] Aktualizace probehne, az odpovis A nebo Y.
pause
exit /b 0

:untarfail
call :cleanup
echo [opt_chp] CHYBA: ZIP se nepodarilo rozbalit.
pause
exit /b 1

:badzip
call :cleanup
echo [opt_chp] CHYBA: v ZIPu neni app.py - je to spravny archiv?
pause
exit /b 1

:badsrc
echo [opt_chp] CHYBA: v "%SRC%" neni app.py - to neni slozka s projektem.
pause
exit /b 1

:baddest
echo [opt_chp] CHYBA: v "%OPT_CHP_HOME%" neni app.py.
echo [opt_chp] Nastav OPT_CHP_HOME na slozku s kodem.
pause
exit /b 1

:failed
call :cleanup
echo [opt_chp] CHYBA: robocopy selhal.
pause
exit /b 1
