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

	prevRootT.setIdentity();

}

const dart::dynamics::SkeletonPtr&
Character3D::
getSkeleton()
{
	return mSkeleton;
}

void 
Character3D::
copy(Character3D *c3d)
{
	this->mSkeleton->setPositions(c3d->mSkeleton->getPositions());
	this->mSkeleton->setVelocities(c3d->mSkeleton->getVelocities());
	this->mName = c3d->mName;

	this->curLeftFingerPosition = c3d->curLeftFingerPosition;
	this->curRightFingerPosition = c3d->curRightFingerPosition;
	this->curLeftFingerBallPosition = c3d->curLeftFingerBallPosition;
	this->curRightFingerBallPosition = c3d->curRightFingerBallPosition;

	this->blocked = c3d->blocked;

	this->inputActionType = c3d->inputActionType;
	this->availableActionTypes = c3d->availableActionTypes;
	this->prevSkelPositions = c3d->prevSkelPositions;
	this->prevKeyJointPositions = c3d->prevKeyJointPositions;
	this->prevRootT = c3d->prevRootT;
}
