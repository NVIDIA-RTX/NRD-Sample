@echo off

git submodule update --init --recursive

mkdir "_Build"

cd "_Build"

cmake .. %*
if %ERRORLEVEL% NEQ 0 exit /B %ERRORLEVEL%

cd ..
