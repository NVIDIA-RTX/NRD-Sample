#!/bin/bash

rm -rf "build"

rm -rf "_Bin"
rm -rf "_Build"
rm -rf "_Data"
rm -rf "_Shaders"
rm -rf "_NRD_SDK"
rm -rf "_NRI_SDK"

cd "External/NRIFramework"
source "4-Clean.sh"
cd "../.."

cd "External/NRD"
source "4-Clean.sh"
cd "../.."
