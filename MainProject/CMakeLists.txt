cmake_minimum_required(VERSION 2.8.12)

project(CV_project)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

# Executables
add_executable(LabelHistograms
    src/LabelHistograms.cpp
)

add_executable(main
    src/main.cpp
)

add_executable(mainTest
    src/mainTest.cpp
)

# Libraries
add_library(HistogramComparator
    include/HistogramComparator.h
    src/HistogramComparator.cpp
)

add_library(FoodFinder
    include/FoodFinder.h
    src/FoodFinder.cpp
)

add_library(MetricsCalculation
    include/MetricsCalculation.h
    src/MetricsCalculation.cpp
)

add_library(FoodSegmenter
    include/FoodSegmenter.h
    src/FoodSegmenter.cpp
)

add_library(Test
    include/Test.h
    src/Test.cpp
)

add_library(Tray
    include/Tray.h
    src/Tray.cpp
)

add_library(Utils
    include/Utils.h
    src/Utils.cpp
)

# Link libraries
target_link_libraries(HistogramComparator ${OpenCV_LIBS})
target_link_libraries(FoodFinder ${OpenCV_LIBS})
target_link_libraries(MetricsCalculation ${OpenCV_LIBS})
target_link_libraries(FoodSegmenter ${OpenCV_LIBS})
target_link_libraries(Test ${OpenCV_LIBS})

target_link_libraries(LabelHistograms HistogramComparator)
target_link_libraries(main Tray)
target_link_libraries(mainTest Tray Test)
target_link_libraries(FoodSegmenter HistogramComparator Utils)
target_link_libraries(Test Tray MetricsCalculation)
target_link_libraries(Tray FoodFinder FoodSegmenter Utils)

# Create data folder
file(MAKE_DIRECTORY data)

# Create main output folder
file(MAKE_DIRECTORY output)

# Create the folder for files
file(MAKE_DIRECTORY output/tray1)
file(MAKE_DIRECTORY output/tray1/masks)
file(MAKE_DIRECTORY output/tray1/bounding_boxes)
file(MAKE_DIRECTORY output/tray2)
file(MAKE_DIRECTORY output/tray2/masks)
file(MAKE_DIRECTORY output/tray2/bounding_boxes)
file(MAKE_DIRECTORY output/tray3)
file(MAKE_DIRECTORY output/tray3/masks)
file(MAKE_DIRECTORY output/tray3/bounding_boxes)
file(MAKE_DIRECTORY output/tray4)
file(MAKE_DIRECTORY output/tray4/masks)
file(MAKE_DIRECTORY output/tray4/bounding_boxes)
file(MAKE_DIRECTORY output/tray5)
file(MAKE_DIRECTORY output/tray5/masks)
file(MAKE_DIRECTORY output/tray5/bounding_boxes)
file(MAKE_DIRECTORY output/tray6)
file(MAKE_DIRECTORY output/tray6/masks)
file(MAKE_DIRECTORY output/tray6/bounding_boxes)
file(MAKE_DIRECTORY output/tray7)
file(MAKE_DIRECTORY output/tray7/masks)
file(MAKE_DIRECTORY output/tray7/bounding_boxes)
file(MAKE_DIRECTORY output/tray8)
file(MAKE_DIRECTORY output/tray8/masks)
file(MAKE_DIRECTORY output/tray8/bounding_boxes)