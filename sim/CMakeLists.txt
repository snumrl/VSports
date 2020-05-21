cmake_minimum_required(VERSION 3.9)
project(sim)
set(CMAKE_CXX_COMPILER /usr/bin/g++-7)

add_compile_options(-fPIC)
#add_compile_options(-std=gnu++11)
add_compile_options(-Wdeprecated-declarations)
add_compile_options(-DHAVE_CSTDDEF)

SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
#	FORCE
	)
find_package(Eigen3 REQUIRED)
find_package(DART REQUIRED COMPONENTS gui optimizer-ipopt CONFIG)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR})


include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB srcs "*.h" "*.cpp" "../render/*.cpp" "../render/*.h" "../utils/*.cpp" "../utils/*.h" "../motion/*.h" "../motion/*.cpp")

add_library(sim ${srcs})
target_link_libraries(sim ${DART_LIBRARIES} ${EIGEN_LIBRARIES} ${Boost_LIBRARIES})
