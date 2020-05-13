#include "Character3D.h"
#include "../model/SkelMaker.h"
#include <dart/dart.hpp>
#include <iostream>

Character3D::
Character3D(const std::string& name)
:mName(name)
{
}

const dart::dynamics::SkeletonPtr&
Character3D::
getSkeleton()
{
	return mSkeleton;
}