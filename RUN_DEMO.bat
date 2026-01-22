@echo off
echo ============================================
echo   TENNIS RACKET 6DOF TRACKING DEMO
echo ============================================
echo.
echo Starting split-screen visualization...
echo.
echo Controls:
echo   SPACE = Pause/Play
echo   A     = Previous frame
echo   D     = Next frame  
echo   Q     = Quit
echo.
echo ============================================
echo.

cd /d "%~dp0"
python split_viewer.py

pause

