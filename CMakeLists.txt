cmake_minimum_required(VERSION 2.8)
project(exercise_4)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
set(CMAKE_CXX_FLAGS "-std=c++14 ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_EXTENSIONS ON)
add_definitions(-DPROJECT_DIR="${PROJECT_SOURCE_DIR}")

# Eigen
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# FreeImage
find_package(FreeImage REQUIRED)
include_directories(${FREEIMAGE_INCLUDE_DIRS})

# Flann
find_package(Flann REQUIRED)
include_directories(${FLANN_INCLUDE_DIRS})

# Ceres Solver
find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})

# OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OPENCV_INCLUDE_DIRS})

# LZ4
find_package(PkgConfig REQUIRED)
pkg_check_modules(LZ4 REQUIRED liblz4)

# PCL
find_package(PCL REQUIRED )
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_subdirectory(./dlib dlib_build)


# Sohpus
include_directories(sophus)

# Set files to be compiled
set(HEADER_FILES
    Eigen.h 
    SimpleMesh.h 
    PointCloud.h 
    VirtualSensor.h 
    NearestNeighbor.h 
    ProcrustesAligner.h 
    ICPOptimizer.h 
    FreeImageHelper.h
    BaselModelReader.h
    ModelFittingOptimizer.h
    RGBDDataLoader.h

)

set(SOURCE_FILES 
    FreeImageHelper.cpp
)

add_executable(final main_final4z.cpp ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(final ${FREEIMAGE_LIBRARIES} ${FLANN_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${LZ4_LIBRARIES}  ${PCL_LIBRARIES} dlib::dlib)

#add_executable(procrustes main.cpp ${HEADER_FILES} ${SOURCE_FILES})
#target_link_libraries(procrustes ${FREEIMAGE_LIBRARIES} ${FLANN_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${LZ4_LIBRARIES} dlib::dlib)
