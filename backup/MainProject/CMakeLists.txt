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

add_library(Trays
    include/Trays.h
    src/Trays.cpp
)


# link libraries
target_link_libraries(PlatesFinder ${OpenCV_LIBS})
target_link_libraries(Trays PlatesFinder)
target_link_libraries(main Trays)

