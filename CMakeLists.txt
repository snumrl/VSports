cmake_minimum_required(VERSION 2.8)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7)

add_compile_options(-fPIC)
#add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)
project(vsports_multi)

SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release."
	)

find_package(Eigen3 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(DART REQUIRED COMPONENTS gui optimizer-ipopt CONFIG)
find_package(Boost 1.66.0 COMPONENTS filesystem python3 numpy3)
find_package(PythonLibs REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${OPENGL_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_DIR})


add_subdirectory( sim )
add_subdirectory( model )
add_subdirectory( pyvs )


file(GLOB srcs
		"./*.cpp"
		"render/*.h" "render/*.cpp"
		"vsports/*.h" "vsports/*.cpp"
		"vsports/common/*.h" "vsports/common/*.cpp")

add_executable(vsports_multi ${srcs})
target_link_libraries(vsports_multi ${OPENGL_LIBRARIES} ${GLUT_LIBRARIES} ${GLEW_LIBRARIES} ${DART_LIBRARIES} ${Boost_LIBRARIES} ${PYTHON_LIBRARIES} sim pyvs model)
