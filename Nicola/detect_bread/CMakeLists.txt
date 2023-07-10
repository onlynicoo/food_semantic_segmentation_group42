project(CV_project)
find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

add_executable(detect_bread
    src/detect_bread.cpp
)

add_library(PlatesFinder
    include/PlatesFinder.h
    src/PlatesFinder.cpp
)

#target_link_libraries(tray_lib ${OpenCV_LIBS})
#target_link_libraries(test_lib ${OpenCV_LIBS})
#target_link_libraries(test01 ${OpenCV_LIBS})
#target_link_libraries(test_lib tray_lib)

#target_link_libraries(asd ${OpenCV_LIBS})
target_link_libraries(PlatesFinder ${OpenCV_LIBS})
target_link_libraries(detect_bread ${OpenCV_LIBS})
target_link_libraries(detect_bread PlatesFinder)
#target_link_libraries(test_lib ${OpenCV_LIBS})