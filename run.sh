#!/bin/bash
mkdir -p guolao
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build