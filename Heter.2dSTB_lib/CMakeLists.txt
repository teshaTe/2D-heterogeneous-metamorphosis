cmake_minimum_required(VERSION 3.1)

project(2D_HSTB_LIB)

set (SRC_HEADERS  include/distanceTransform2D.h
                  include/interpolation2D.h
                  include/heter2dSTB.h
                  include/imageProc.h
                  include/STTI.h
                  include/timer.hpp
    )

set (SRC_SOURCE   src/distanceTransform2D.cpp
                  src/interpolation2D.cpp
                  src/heter2dSTB.cpp
                  src/imageProc.cpp
                  src/STTI.cpp
    )

add_library(hstb_lib ${SRC_SOURCE} ${SRC_HEADERS})
target_link_libraries(hstb_lib  opencv_imgcodecs opencv_highgui opencv_imgproc opencv_core gtmathematicsgpu)

if(USE_OPENMP)
    target_link_libraries(hstb_lib gomp omp omptarget)
endif()
