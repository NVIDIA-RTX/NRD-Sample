#!/bin/bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")/../.."

git submodule update --init --recursive

cmake -S . -B _Build "$@"
