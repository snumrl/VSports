cmake_minimum_required(VERSION 3.9)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7)

add_compile_options(-fPIC)
add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)
project(vsports)

#SET(CMAKE_BUILD_TYPE Release CACHE STRING
#	"Choose the type of build, options are: None Debug Release."
#	)


if (NOT EXISTS ${CMAKE_BINARY_DIR}/CMakeCache.txt)
  if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
  endif()
endif()


add_definitions(-DVSPORTS_HOME_DIR="${CMAKE_HOME_DIRECTORY}")

find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
#find_package(GLEW REQUIRED)
find_package(Boost 1.73.0 COMPONENTS filesystem python3 numpy3)
find_package(DART REQUIRED COMPONENTS gui optimizer-ipopt CONFIG )
find_package(PythonLibs REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
#include_directories(${GLEW_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})
include_directories(plugin/ICA)
include_directories(extern)


add_subdirectory( extern/ICA/plugin )
add_subdirectory( sim )
add_subdirectory( model )
add_subdirectory( pyvs )


#file(GLOB srcs
#		"./*.cpp"
#		"render/*.h" "render/*.cpp"
#		"vsports/*.h" "vsports/*.cpp"
#		"utils/*.h" "utils/*.cpp"
#		"motion/*.h" "motion/*.cpp")

file(GLOB srcs
		"./*.cpp"
		"render/*.h" "render/*.cpp"
		"vsports/common.h"
		#"vsports/SingleBasketballWindow.h" "vsports/SingleBasketballWindow.cpp"
		"vsports/SingleControlWindow.h" "vsports/SingleControlWindow.cpp"
		"vsports/MultiHeadWindow.h" "vsports/MultiHeadWindow.cpp"
		"utils/*.h" "utils/*.cpp"
		"motion/*.h" "motion/*.cpp")

add_executable(vsports ${srcs})
target_link_libraries(vsports ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${DART_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} sim pyvs model plugin_ICA)