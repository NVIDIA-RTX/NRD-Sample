@echo off

git submodule update --init --recursive

mkdir "_Build"

cd "_Build"

cmake .. %*
if %ERRORLEVEL% NEQ 0 goto END

:END
cd ..
exit /B %ERRORLEVEL%
