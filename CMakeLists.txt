cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

#set standard of C++11 and build
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

project(camera_fusion)

find_package(OpenCV 4.1 REQUIRED)
#Add plotter dub-directory
add_subdirectory(plotter)
list(APPEND EXTRA_LIBS plotter)
list(APPEND EXTRA_INCLUDES "${PROJECT_SOURCE_DIR}/plotter")

include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})
list(APPEND EXTRA_LIBS "${OpenCV_LIBRARIES}")
list(APPEND EXTRA_INCLUDES "${OpenCV_LIBRARY_DIRS}")

# Executable for create matrix exercise
add_executable (3D_object_tracking src/camFusion_Student.cpp src/FinalProject_Camera.cpp src/lidarData.cpp src/matching2D_Student.cpp src/objectDetection2D.cpp)
target_include_directories(3D_object_tracking PUBLIC ${EXTRA_INCLUDES})
target_link_libraries (3D_object_tracking ${EXTRA_LIBS})