cmake_minimum_required(VERSION 2.8.6)
project(pyvs)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)

find_package(PythonLibs REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB srcs "*.h" "*.cpp")

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_library(pyvs SHARED ${srcs})

target_link_libraries(pyvs ${DART_LIBRARIES} ${Boost_LIBRARIES} ${EIGEN_LIBRARIES} ${PYTHON_LIBRARIES} sim model)
set_target_properties(pyvs PROPERTIES PREFIX "" )