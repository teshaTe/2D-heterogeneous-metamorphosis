cmake_minimum_required(VERSION 3.1)

project(2DSTB_LIB)

set (SRC_HEADERS  include/distanceTransform2D.h
                  include/metamorphosis.h
                  include/imageProc.h
                  include/colourSegmentation.h
                  include/colourInterpolation.h
    )

set (SRC_SOURCE   src/distanceTransform2D.cpp
                  src/metamorphosis.cpp
                  src/imageProc.cpp
                  src/colourSegmentation.cpp
                  src/colourInterpolation.cpp
    )

add_library(stb_lib ${SRC_SOURCE} ${SRC_HEADERS})
target_link_libraries(stb_lib  opencv_imgcodecs opencv_highgui
                               opencv_imgproc opencv_core )

if(USE_OPENMP)
    target_link_libraries(stb_lib gomp omp omptarget)
endif()
