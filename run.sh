#!/bin/bash
mkdir -p guolao
touch -c guolao/__init__.py
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build