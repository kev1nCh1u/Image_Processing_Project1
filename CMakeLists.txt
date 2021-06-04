cmake_minimum_required(VERSION 2.8)
project( ImageProject1 )
set(OpenCV_DIR /home/user/opencv/opencv3/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( ImageProject1 src/demo_code.cpp )
target_link_libraries( ImageProject1 ${OpenCV_LIBS} )

add_executable( cv_version src/cv_version.cpp )
target_link_libraries( cv_version ${OpenCV_LIBS} )