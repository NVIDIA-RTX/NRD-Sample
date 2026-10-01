#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

rm -rf "build"
rm -rf "_Bin"
rm -rf "_Build"
rm -rf "_Data"
rm -rf "_Shaders"
rm -rf "_NRD_SDK"
rm -rf "_NRI_SDK"

bash "External/NRIFramework/Scripts/Linux/4-Clean.sh"
(cd "External/NRD" && bash -e "4-Clean.sh")
