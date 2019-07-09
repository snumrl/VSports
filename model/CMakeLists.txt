cmake_minimum_required(VERSION 2.8.6)
project(model)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)
find_package(DART REQUIRED COMPONENTS gui optimizer-ipopt CONFIG)

file(GLOB srcs "*.h" "*.cpp")

include(FindOpenMP)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_library(model SHARED ${srcs})

target_link_libraries(model ${DART_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
set_target_properties(model PROPERTIES PREFIX "" )