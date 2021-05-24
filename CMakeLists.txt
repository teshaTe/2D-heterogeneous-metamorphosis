cmake_minimum_required(VERSION 3.1)

project(2d_blending)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(PROJECT_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

option(USE_DEBUG_INFO   "Added picture output with circumscribed circles/rectangles" OFF)
option(USE_OPENMP       "Use OpenMP for parallel calculation (optional)"             OFF)
option(BUILD_FOR_WINDOWS    "Build project for the Microsoft Visual Studio (optional)"   OFF)

set(CMAKE_CXX_STANDARD 11)

if (BUILD_FOR_WINDOWS)
    # define for the Windows builds
    set(VCPKG_ROOT "")
    if(VCPKG_ROOT STREQUAL "")
        message(FATAL_ERROR "ERROR: no VCPKG_ROOT specified in the root cmake list!")
    else()
        message("The vcpkg root directory is: ${VCPKG_ROOT}")
    endif()
    set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
else()
    set(CMAKE_CXX_FLAGS " -pthread ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
endif()

if(USE_DEBUG_INFO)
    add_definitions(-DUSE_DEBUG_INFO)
endif()

if(USE_OPENMP)
    find_package(OpenMP REQUIRED)
    add_definitions(-DUSE_OPENMP)
    if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
        if(BUILD_FOR_WINDOWS)
            set(CMAKE_CXX_FLAGS "-Xclang -openmp ${CMAKE_CXX_FLAGS}")
        else()
            set(CMAKE_CXX_FLAGS "-fopenmp=libomp ${CMAKE_CXX_FLAGS}")
        endif()
    elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
        set(CMAKE_CXX_FLAGS "-qopenmp ${CMAKE_CXX_FLAGS}")
    else()
        set(CMAKE_CXX_FLAGS "-fopenmp ${CMAKE_CXX_FLAGS}")
    endif()
endif(USE_OPENMP)

find_package(glm REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SFML REQUIRED)

include_directories(${GLM_INCLUDE_DIRS})
include_directories(${SFML_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/Heter.2dSTB_lib")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/ExternalLibs/GeoTools")

add_subdirectory(Heter.2dSTB_lib)
add_subdirectory(samples)
