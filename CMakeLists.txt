cmake_minimum_required(VERSION 2.8)
project( ImageProject1 )

set(OpenCV_DIR /home/user/opencv/opencv3/build)
# set(OpenCV_DIR /home/kevin/lib/opencv3.4.14/build)
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( Eigen3 REQUIRED )
MESSAGE( [Main] " EIGEN3_INCLUDE_DIRS = ${EIGEN3_INCLUDE_DIRS}") 
# EIGEN3_INCLUDE_DIRS = /usr/include/eigen3
include_directories(${EIGEN3_INCLUDE_DIRS})

add_executable( ImageProject1 src/demo_code.cpp )
target_link_libraries( ImageProject1 ${OpenCV_LIBS} )

add_executable( cv_version src/cv_version.cpp )
target_link_libraries( cv_version ${OpenCV_LIBS} )