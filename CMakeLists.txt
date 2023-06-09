find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

add_executable( test01
    src/test01.cpp
)

target_link_libraries(utilities_functions ${OpenCV_LIBS})