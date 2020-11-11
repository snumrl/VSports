#include "Character3D.h"
#include "../model/SkelMaker.h"
#include <dart/dart.hpp>
#include <iostream>

Character3D::
Character3D(const std::string& name)
:mName(name)
{
	curLeftFingerPosition = 0.0;
	curRightFingerPosition = 0.0;
	curLeftFingerBallPosition = 0.0;
	curRightFingerBallPosition = 0.0;
	blocked = false;
	inputActionType = 0;
	availableActionTypes.resize(2);

	// prevSkelPositions.resize(mSkeleton->getNumDofs());
	// prevSkelPositions
	prevKeyJointPositions.resize(6*3);
	prevKeyJointPositions.setZero();

}

const dart::dynamics::SkeletonPtr&
Character3D::
getSkeleton()
{
	return mSkeleton;
}