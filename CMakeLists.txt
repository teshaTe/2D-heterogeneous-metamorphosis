cmake_minimum_required(VERSION 3.1)

project(2d_blending)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-O3 -pthread ${CMAKE_CXX_FLAGS}")

option(USE_SMOOTHED_CYLINDERS       OFF)
option(USE_DISTANCE_FIELDS           ON)
option(USE_AFFINE_TRANSFORMATIONS   OFF)
option(USE_CONE_F3                  OFF)
option(USE_PYRAMID_F3               OFF)
option(USE_EGG_SHAPE_F3             OFF)
option(USE_TRANSPARENT_BACKBGROUND  OFF)
option(USE_CIRCLE_CIRCLE->CROSS     OFF)
option(USE_CIRCLE_RING->CROSS       OFF)
option(USE_CIRCLE1->CIRCLE2         OFF)
option(USE_OPENMP                   ON)

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

if(USE_EGG_SHAPE_F3)
  add_definitions(-DUSE_EGG_SHAPE_F3)
endif()

if(USE_TRANSPARENT_BACKBGROUND)
  add_definitions(-DUSE_TRANSPARENT_BACKBGROUND)
endif()

if(USE_CIRCLE_CIRCLE->CROSS)
  add_definitions(-DUSE_CIRCLE_CIRCLE_CROSS)
endif()

if(USE_CIRCLE_RING->CROSS)
  add_definitions(-DUSE_CIRCLE_RING_CROSS)
endif()

if(USE_CIRCLE1->CIRCLE2)
  add_definitions(-DUSE_CIRCLE_CIRCLE)
endif()

if(USE_OPENMP)
    add_definitions(-DUSE_OPENMP)
    find_package(OpenMP)
    set(CMAKE_CXX_FLAGS " -fopenmp ${CMAKE_CXX_FLAGS}")
endif()

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/NGL/include")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/NGL/lib")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/opencv/include")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/opencv/lib")

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
target_link_libraries(2d_blend NGL boost_system opencv_highgui opencv_core opencv_imgcodecs opencv_imgproc ${OpenMP_LIBRARIES} Qt5::OpenGL Qt5::Widgets)
