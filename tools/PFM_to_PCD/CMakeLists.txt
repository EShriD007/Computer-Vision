
cmake_minimum_required(VERSION 3.5.1)
project(PFMTOPCDCONVERTER)
include(${CMAKE_ROOT}/Modules/ExternalProject.cmake)

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

set(OUTPUT "PFMtoPCD")
set(CMAKE_BUILD_TYPE Release)

find_package(OpenCV REQUIRED )
find_package(PCL 1.10 REQUIRED )

set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations -std=c++17")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
)

link_directories(${PCL_LIBRARY_DIRS})

MESSAGE(STATUS "PROJECT BINARY: ${PROJECT_BINARY_DIR}")
MESSAGE(STATUS "PCL LIBRARY   : ${PCL_LIBRARY_DIRS}")
MESSAGE(STATUS "PCL INCLUDE   : ${PCL_INCLUDE_DIRS}")
MESSAGE(STATUS "OPENCV INCLUDE: ${OpenCV_INCLUDE_DIRS}")
MESSAGE(STATUS "OPENCV Version: ${OpenCV_VERSION}")
MESSAGE(STATUS "PCL Version   : ${PCL_VERSION}")

add_definitions(${PCL_DEFINITIONS})
set_directory_properties( PROPERTIES COMPILE_DEFINITIONS "" )
add_executable(PFMtoPCD pfm2pcd.cpp)

target_link_libraries(PFMtoPCD
			${OpenCV_LIBS}
			${PCL_LIBRARIES}
			config++
			)

install(DIRECTORY  ${PROJECT_BINARY_DIR}/bin/ USE_SOURCE_PERMISSIONS DESTINATION /usr/bin)