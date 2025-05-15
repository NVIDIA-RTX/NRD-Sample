@echo off

if exist "build" rd /q /s "build"

if exist "_Bin" rd /q /s "_Bin"
if exist "_Build" rd /q /s "_Build"
if exist "_Data" rd /q /s "_Data"
if exist "_Shaders" rd /q /s "_Shaders"

cd "External/NRIFramework"
call "4-Clean.bat"
cd "../.."

cd "External/NRD"
call "4-Clean.bat"
cd "../.."
