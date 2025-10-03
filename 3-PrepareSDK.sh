#!/bin/bash

ROOT=$(pwd)
NRD=$(ROOT)/External/NRD
NRI=$(ROOT)/External/NRIFramework/External/NRI

rm -rf "_NRD_SDK"
mkdir -p "_NRD_SDK"
cd "_NRD_SDK"

mkdir -p "Include"
mkdir -p "Integration"
mkdir -p "Lib/Debug"
mkdir -p "Lib/Release"
mkdir -p "Shaders"

cp -r "$(NRD)/Include/" "Include"
cp -r "$(NRD)/Integration/" "Integration"
cp -r "$(NRD)/Shaders/Include/NRD.hlsli" "Shaders"
cp -r "$(NRD)/Shaders/Include/NRDConfig.hlsli" "Shaders"
cp "$(NRD)/LICENSE.txt" "."
cp "$(NRD)/README.md" "."
cp "$(NRD)/UPDATE.md" "."

cp -H "$(ROOT)/_Bin/Debug/libNRD.so" "Lib/Debug"
cp -H "$(ROOT)/_Bin/Release/libNRD.so" "Lib/Release"

cd ..

rm -rf "_NRI_SDK"
mkdir -p "_NRI_SDK"
cd "_NRI_SDK"

mkdir -p "Include"
mkdir -p "Lib/Debug"
mkdir -p "Lib/Release"

cp -r "$(NRI)/Include/" "Include"
cp "$(NRI)/LICENSE.txt" "."
cp "$(NRI)/README.md" "."
cp "$(NRI)/nri.natvis" "."

cp -H "$(ROOT)/_Bin/Debug/libNRI.so" "Lib/Debug"
cp -H "$(ROOT)/_Bin/Release/libNRI.so" "Lib/Release"

cd ..
