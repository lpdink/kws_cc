cmake_minimum_required(VERSION 3.5)

project(crn_test)

set(CMAKE_CXX_FLAGS " -O3 -fopenmp -std=c++11 ")
set(CMAKE_C_FLAGS " -O3 -fopenmp -std=c++11 ")

add_executable(crn_test
               ./src/crn_test.cc)

target_include_directories(crn_test PUBLIC ./inc)