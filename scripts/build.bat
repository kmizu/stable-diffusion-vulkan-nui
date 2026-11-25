@echo off
setlocal
cd /d "%~dp0.."
call sbt -batch ui/assembly vulkanWorker/assembly
exit /b %ERRORLEVEL%
