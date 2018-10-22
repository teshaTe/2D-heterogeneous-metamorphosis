cmake_minimum_required(VERSION 3.1)

project(2d_blending)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS " -pthread ${CMAKE_CXX_FLAGS}")

option(USE_DEBUG_INFO               OFF)
option(USE_SMOOTHED_CYLINDERS       ON)
option(USE_DISTANCE_FIELDS          ON)
option(USE_AFFINE_TRANSFORMATIONS   OFF)
option(USE_CONE_F3                  OFF)
option(USE_PYRAMID_F3               OFF)
option(USE_TRANSPARENT_BACKBGROUND  OFF)
option(USE_OPENMP                   OFF)
option(USE_CROSS_DISSOLVE           OFF)

if(USE_DEBUG_INFO)
    add_definitions(-DUSE_DEBUG_INFO)
endif()

if(USE_SMOOTHED_CYLINDERS)
    add_definitions(-DUSE_SMOOTHED_CYLINDERS)
endif()

if(USE_DISTANCE_FIELDS)
  add_definitions(-DUSE_DISTANCE_FIELD)
endif()

if(USE_AFFINE_TRANSFORMATIONS)
  add_definitions(-DUSE_AFFINE_TRANSFORMATIONS)
endif()

if(USE_CONE_F3)
  add_definitions(-DUSE_CONE_F3)
endif()

if(USE_PYRAMID_F3)
  add_definitions(-DUSE_PYRAMID_F3)
endif()

if(USE_TRANSPARENT_BACKBGROUND)
  add_definitions(-DUSE_TRANSPARENT_BACKBGROUND)
endif()

if(USE_OPENMP)
    add_definitions(-DUSE_OPENMP)

    include_directories("${CMAKE_CURRENT_SOURCE_DIR}/openmp/include")
    link_directories("${CMAKE_CURRENT_SOURCE_DIR}/openmp/lib")

    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "-fopenmp=libomp ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
    endif()
endif(USE_OPENMP)

if(USE_CROSS_DISSOLVE)
    add_definitions(-DUSE_CROSS_DISSOLVE)
endif()


include_directories("${CMAKE_CURRENT_SOURCE_DIR}/NGL/include")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/NGL/lib")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/opencv/include")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/opencv/clang_lib")

find_package(Qt5OpenGL)

set (SRC_HEADERS
        include/distance_field.h
        include/shader.h
        include/stb_tricks.h
    )

set (SRC_SOURCE
        src/2d_blend.cpp
        src/distance_field.cpp
        src/shader.cpp
        src/stb_tricks.cpp
    )

add_executable(2d_blend src/2d_blend.cpp ${SRC_SOURCE} ${SRC_HEADERS})
target_link_libraries(2d_blend NGL boost_system opencv_imgcodecs opencv_highgui
                               opencv_imgproc opencv_core Qt5::OpenGL Qt5::Widgets)

if(USE_OPENMP)
    target_link_libraries(2d_blend gomp omp omptarget omptarget-nvptx omptarget.rtl.cuda)
endif()
