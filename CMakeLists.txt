cmake_minimum_required(VERSION 3.19)
project(tracer)

set(CMAKE_CXX_STANDARD 17)
include_directories({MPI_INCLUDE_PATH})
set(CMAKE_CXX_COMPILER "/usr/bin/mpic++")

file(GLOB SOURCES
    src/*.*
)
add_definitions(-DSTB_IMAGE_WRITE_IMPLEMENTATION)
add_executable(tracer ${SOURCES})
