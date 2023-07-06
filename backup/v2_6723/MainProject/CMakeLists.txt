cmake_minimum_required(VERSION 2.8.12)

project(CV_project)

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

# Executables
add_executable(main
    src/main.cpp
)

add_executable(LabelFeatures
    src/LabelFeatures.cpp
)

# Libraries
add_library(PlatesFinder
    include/PlatesFinder.h
    src/PlatesFinder.cpp
)

add_library(Tray
    include/Tray.h
    src/Tray.cpp
)

add_library(PlateRemover
    include/PlateRemover.h
    src/PlateRemover.cpp
)

add_library(FeatureComparator
    include/FeatureComparator.h
    src/FeatureComparator.cpp
)

# Link libraries
target_link_libraries(PlatesFinder ${OpenCV_LIBS})
target_link_libraries(PlateRemover ${OpenCV_LIBS})
target_link_libraries(FeatureComparator ${OpenCV_LIBS})
target_link_libraries(Tray PlatesFinder PlateRemover FeatureComparator)
target_link_libraries(main Tray)
target_link_libraries(LabelFeatures FeatureComparator)