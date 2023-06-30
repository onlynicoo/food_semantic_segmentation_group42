cmake_minimum_required(VERSION 2.8.12)

project(food)

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))
include_directories(include)

add_executable(food
    src/food.cpp
    include/plates.cpp
    include/plates.h
    include/FeatureComparator.cpp
    include/FeatureComparator.h)

target_link_libraries(food ${OpenCV_LIBS})