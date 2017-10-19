cmake_minimum_required(VERSION 3.8)
project(rmpe_dataset_server)

set(CMAKE_CXX_STANDARD 11)

find_package(HDF5 REQUIRED COMPONENTS C CXX)
find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS thread filesystem system REQUIRED)
find_package(cppzmq REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES (${HDF5_INCLUDE_DIRS})
include_directories(${cppzmq_INCLUDE_DIR})

set(SOURCE_FILES main.cpp DataTransformer.cpp DataTransformer.h RNGen.cpp RNGen.h utils.h)
add_executable(rmpe_dataset_server ${SOURCE_FILES})

target_link_libraries(rmpe_dataset_server
        ${OpenCV_LIBS}
        ${Boost_FILESYSTEM_LIBRARY}
        ${Boost_THREAD_LIBRARY}
        ${Boost_SYSTEM_LIBRARY}
        ${HDF5_CXX_LIBRARIES}
        ${HDF5_LIBRARIES}
        ${cppzmq_LIBRARY}
        )
