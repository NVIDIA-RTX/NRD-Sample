#!/bin/bash
set -e

git submodule update --init --recursive

mkdir -p "_Build"

cd "_Build"
cmake .. "$@"
cd ..
