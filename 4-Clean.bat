@echo off

if exist "build" rd /q /s "build"

if exist "_Bin" rd /q /s "_Bin"
if exist "_Build" rd /q /s "_Build"
if exist "_Data" rd /q /s "_Data"
if exist "_Shaders" rd /q /s "_Shaders"
if exist "_NRD_SDK" rd /q /s "_NRD_SDK"
if exist "_NRI_SDK" rd /q /s "_NRI_SDK"

cd "External/NRIFramework"
call "4-Clean.bat"
cd "../.."

cd "External/NRD"
call "4-Clean.bat"
cd "../.."
