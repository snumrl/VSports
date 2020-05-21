cmake_minimum_required(VERSION 3.9)
project(pyvs)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)

find_package(PythonLibs REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

file(GLOB srcs "*.h" "*.cpp")

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})


set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_library(pyvs SHARED ${srcs})

target_link_libraries(pyvs ${DART_LIBRARIES} ${Boost_LIBRARIES} ${EIGEN_LIBRARIES} ${PYTHON_LIBRARIES} sim model plugin_ICA OpenMP::OpenMP_CXX)
set_target_properties(pyvs PROPERTIES PREFIX "" )