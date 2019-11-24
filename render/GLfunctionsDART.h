#ifndef __GL_FUNCTIONS_DART_H__
#define __GL_FUNCTIONS_DART_H__
#include "dart/dart.hpp"
#include <GL/glew.h>
// #include "dart/gui/gui.hpp"
#include "dart/math/math.hpp"
#include "dart/simulation/simulation.hpp"
#include "GLfunctions.h"

namespace GUI
{
void drawSkeleton(
		const dart::dynamics::SkeletonPtr& skel,
		const Eigen::Vector3d& color=Eigen::Vector3d(0.8,0.8,0.8),
		bool wireFrame = false);

void drawShape(const Eigen::Isometry3d& T,
	const dart::dynamics::Shape* shape,
	const Eigen::Vector3d& color,
	bool wireFrame = false);
}

#endif