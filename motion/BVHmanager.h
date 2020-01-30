#ifndef __BVH_MANAGER_H__
#define __BVH_MANAGER_H__
#include "BVHparser.h"
#include <dart/dart.hpp>
namespace BVHmanager
{
	void setPositionFromBVH(dart::dynamics::SkeletonPtr skel, BVHparser *bvhParser, int bvh_frame);
}
#endif