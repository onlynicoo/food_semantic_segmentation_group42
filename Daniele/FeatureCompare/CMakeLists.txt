cmake_minimum_required(VERSION 2.8.12)

project(features)

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))
include_directories(include)

add_executable(features
    src/compare.cpp
    include/features.cpp
    include/features.h)
target_link_libraries(features ${OpenCV_LIBS})