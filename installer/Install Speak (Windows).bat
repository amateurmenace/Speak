@echo off
rem Speak installer — double-click to install Speak.ofx.bundle (next to this
rem script) into the system-wide OpenFX plugin folder. Self-elevates to admin.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrator access...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

set "SRC=%~dp0Speak.ofx.bundle"
set "DST=C:\Program Files\Common Files\OFX\Plugins\Speak.ofx.bundle"

if not exist "%SRC%" (
    echo Speak.ofx.bundle not found next to this script.
    pause
    exit /b 1
)

if not exist "C:\Program Files\Common Files\OFX\Plugins" mkdir "C:\Program Files\Common Files\OFX\Plugins"
if exist "%DST%" rmdir /s /q "%DST%"
xcopy /E /I /Y "%SRC%" "%DST%" >nul

if exist "%DST%" (
    echo.
    echo Installed. Restart DaVinci Resolve, then find it under:
    echo   Color page - Effects - OpenFX - Filters - Hush - Speak Film
) else (
    echo Installation failed.
)
pause
