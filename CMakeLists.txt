# Matches supporting modern CUDA
cmake_minimum_required(VERSION 3.18...4.2.3)

# CUDA will be required (later)
project(
    path_planning_agent
    VERSION 0.1.0
    LANGUAGES CXX CUDA
)

# find pybind11 and Python
if(NOT pybind11_DIR)
    execute_process(
        COMMAND python3 -m pybind11 --cmakedir
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
endif()

find_package(pybind11 CONFIG REQUIRED)

# direct all output compiled binaries into guolao 
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/guolao)

pybind11_add_module(runge_kutta optim/runge_kutta.cpp)

# profiling build flag toggle
option(ENABLE_PROFILING "Include debug symbols and frame pointers for profilers" OFF)
if(ENABLE_PROFILING)
    target_compile_options(runge_kutta PRIVATE -g -fno-omit-frame-pointer)
endif()