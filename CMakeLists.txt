cmake_minimum_required(VERSION 3.1)

project(2d_blending)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS " -pthread ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

option(USE_DEBUG_INFO   "Added picture output with circumscribed circles/rectangles" OFF)
option(USE_OPENMP       "Use OpenMP for parallel calculation (optional)"             OFF)

if(USE_DEBUG_INFO)
    add_definitions(-DUSE_DEBUG_INFO)
endif()

if(USE_OPENMP)
    find_package(OpenMP)
    add_definitions(-DUSE_OPENMP)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        set(CMAKE_CXX_FLAGS "-fopenmp=libomp ${CMAKE_CXX_FLAGS}")
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "-qopenmp ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
    endif()
endif(USE_OPENMP)

find_package(OpenCV REQUIRED)
find_package(glm REQUIRED)

include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/Heter.2dSTB_lib" )
include_directories( "${CMAKE_CURRENT_SOURCE_DIR}/ExternalLibs/GeoTools")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/ExternalLibs/GeoTools/lib")

add_subdirectory(Heter.2dSTB_lib)
add_subdirectory(samples)
