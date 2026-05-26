@echo off
setlocal

REM Define paths
set SCRIPT_DIR=%~dp0
set MOS_BRAIN_DIR=%SCRIPT_DIR%..\..\..\
set SOCCERLAB_DIR=%MOS_BRAIN_DIR%..\soccerLab

REM Check/Set SOCCERLAB_PATH
if "%SOCCERLAB_PATH%"=="" set SOCCERLAB_PATH=%SOCCERLAB_DIR%
echo Using SoccerLab at: %SOCCERLAB_PATH%

REM Launch Sim Server
REM User needs to ensure they are using the python environment compatible with Isaac Lab
REM Example: calling via a specific python executable or conda env
REM For now we just call python.
echo Starting Sim Server...
python "%SCRIPT_DIR%sim_server.py" %*

pause
