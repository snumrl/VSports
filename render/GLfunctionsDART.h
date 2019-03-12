#ifndef __GL_FUNCTIONS_DART_H__
#define __GL_FUNCTIONS_DART_H__
#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include "dart/math/math.hpp"
#include "dart/simulation/simulation.hpp"
#include "GLfunctions.h"

namespace GUI
{
void drawSkeleton(
		const dart::dynamics::SkeletonPtr& skel,
		const Eigen::Vector3d& color=Eigen::Vector3d(0.8,0.8,0.8));

void drawShape(const Eigen::Isometry3d& T,
	const dart::dynamics::Shape* shape,
	const Eigen::Vector3d& color);
}

#endif