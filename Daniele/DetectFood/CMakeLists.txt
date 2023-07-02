cmake_minimum_required(VERSION 2.8.12)

project(DetectFood)

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))
include_directories(include)

add_executable(LabelFeatures
    src/LabelFeatures.cpp
    src/FeatureComparator.cpp    
    include/FeatureComparator.h)

add_executable(DetectFood
    src/DetectFood.cpp
    src/PlateRemover.cpp
    include/PlateRemover.h
    src/FeatureComparator.cpp   
    include/FeatureComparator.h)

target_link_libraries(LabelFeatures ${OpenCV_LIBS})
target_link_libraries(DetectFood ${OpenCV_LIBS})