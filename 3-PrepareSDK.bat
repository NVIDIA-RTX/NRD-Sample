@echo off

set ROOT=%cd%
set NRD=%ROOT%\External\NRD
set NRI=%ROOT%\External\NRIFramework\External\NRI

rd /q /s "_NRD_SDK"
mkdir "_NRD_SDK"
cd "_NRD_SDK"

mkdir "Include"
mkdir "Integration"
mkdir "Lib\Debug"
mkdir "Lib\Release"
mkdir "Shaders"

copy "%NRD%\Include\*" "Include"
copy "%NRD%\Integration\*" "Integration"
copy "%NRD%\Shaders\Include\NRD.hlsli" "Shaders"
copy "%NRD%\Shaders\Include\NRDConfig.hlsli" "Shaders"
copy "%NRD%\LICENSE.txt" "."
copy "%NRD%\README.md" "."
copy "%NRD%\UPDATE.md" "."

copy "%ROOT%\_Bin\Debug\NRD.dll" "Lib\Debug"
copy "%ROOT%\_Bin\Debug\NRD.lib" "Lib\Debug"
copy "%ROOT%\_Bin\Debug\NRD.pdb" "Lib\Debug"
copy "%ROOT%\_Bin\Release\NRD.dll" "Lib\Release"
copy "%ROOT%\_Bin\Release\NRD.lib" "Lib\Release"
copy "%ROOT%\_Bin\Release\NRD.pdb" "Lib\Release"

cd ..

rd /q /s "_NRI_SDK"
mkdir "_NRI_SDK"
cd "_NRI_SDK"

mkdir "Include\Extensions"
mkdir "Lib\Debug"
mkdir "Lib\Release"

copy "%NRI%\Include\*" "Include"
copy "%NRI%\Include\Extensions\*" "Include\Extensions"
copy "%NRI%\LICENSE.txt" "."
copy "%NRI%\README.md" "."
copy "%NRI%\nri.natvis" "."

copy "%ROOT%\_Bin\Debug\NRI.dll" "Lib\Debug"
copy "%ROOT%\_Bin\Debug\NRI.lib" "Lib\Debug"
copy "%ROOT%\_Bin\Debug\NRI.pdb" "Lib\Debug"
copy "%ROOT%\_Bin\Release\NRI.dll" "Lib\Release"
copy "%ROOT%\_Bin\Release\NRI.lib" "Lib\Release"
copy "%ROOT%\_Bin\Release\NRI.pdb" "Lib\Release"

cd ..

