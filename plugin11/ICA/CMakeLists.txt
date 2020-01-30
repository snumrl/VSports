cmake_minimum_required(VERSION 2.8.6)
project(plugin_ICA)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7)

add_compile_options(-fPIC)
#add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)

include_directories(${PYTHON_INCLUDE_DIR})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories( ../../extern )
include_directories(../../extern/ICA/build)


find_package(PythonLibs REQUIRED)
find_package(Eigen3 REQUIRED)

file(GLOB srcs "*.h" "*.cpp")



set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})

add_library(plugin_ICA SHARED ${srcs})

target_link_libraries(plugin_ICA ${DART_LIBRARIES} ${Boost_LIBRARIES} ${EIGEN_LIBRARIES} ${PYTHON_LIBRARIES} 
/home/minseok/Project/VSports/extern/ICA/build/Motion/libMotion.a
/home/minseok/Project/VSports/extern/ICA/build/rnn2/librnn2RunTime.a
/home/minseok/Project/VSports/extern/ICA/build/CharacterControl/libCharacterControl.a
/home/minseok/Project/VSports/extern/ICA/build/Utils/libUtils.a)
set_target_properties(plugin_ICA PROPERTIES PREFIX "" )