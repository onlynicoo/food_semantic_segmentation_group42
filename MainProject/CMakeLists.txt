project(CV_project)
find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

# executables
add_executable(main
    src/main.cpp
)

# libraries
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

# link libraries
target_link_libraries(PlatesFinder ${OpenCV_LIBS})
target_link_libraries(PlateRemover ${OpenCV_LIBS})
target_link_libraries(FeatureComparator ${OpenCV_LIBS})
target_link_libraries(Tray PlatesFinder)
target_link_libraries(Tray PlateRemover)
target_link_libraries(Tray FeatureComparator)
target_link_libraries(main Tray)

