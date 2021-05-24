cmake_minimum_required(VERSION 3.1)

project(2d_blending)

add_executable(2d_blend 2d_blend.cpp )
target_link_libraries(2d_blend opencv_imgcodecs opencv_highgui
                               opencv_imgproc opencv_core hstb_lib)
