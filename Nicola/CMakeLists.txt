project(CV_project)
find_package(OpenCV REQUIRED)
include_directories($(OpenCV_INCLUDE_DIRS))

#add_executable( test01
#    src/test01.cpp
#)

add_executable(test_lib
    src/test_lib.cpp
)

add_library(tray_lib
    include/Tray.h
    src/Tray.cpp
)

#target_link_libraries(tray_lib ${OpenCV_LIBS})
#target_link_libraries(test_lib ${OpenCV_LIBS})
#target_link_libraries(test01 ${OpenCV_LIBS})
#target_link_libraries(test_lib tray_lib)

#target_link_libraries(asd ${OpenCV_LIBS})
target_link_libraries(tray_lib ${OpenCV_LIBS})
target_link_libraries(test_lib tray_lib)
#target_link_libraries(test_lib ${OpenCV_LIBS})