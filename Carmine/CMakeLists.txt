cmake_minimum_required(VERSION 2.8.12)

project(segmentationTask)
find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(ConsoleApplication1
    src/ConsoleApplication1.cpp)

add_library(gaborFeatures
    include/gaborFeatures.h
    src/gaborFeatures.cpp
)
add_library(mySegmentation
    include/mySegmentation.h
    src/mySegmentation.cpp
)

target_link_libraries(gaborFeatures ${OpenCV_LIBS})
target_link_libraries(mySegmentation ${OpenCV_LIBS})
target_link_libraries(mySegmentation gaborFeatures)
target_link_libraries(ConsoleApplication1 mySegmentation)