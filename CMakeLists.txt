cmake_minimum_required(VERSION 3.17)
project(d)

set(CMAKE_CXX_STANDARD 14)

include_directories(${PROJECT_SOURCE_DIR}/include)

link_directories(${PROJECT_SOURCE_DIR}/lib)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mcmodel=large -fopenmp")
add_executable(d main.cpp)
target_link_libraries(d libopenblas.a)