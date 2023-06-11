cmake_minimum_required(VERSION 2.8.12)

project(plates)

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(plates
    src/plates.cpp)
target_link_libraries(plates ${OpenCV_LIBS})