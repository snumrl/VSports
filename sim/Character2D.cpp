#include "Character2D.h"
#include "../model/SkelMaker.h"
#include <dart/dart.hpp>
#include <iostream>

Character2D::
Character2D(const std::string& name)
:mName(name)
{
	this->mSkeleton = dart::dynamics::Skeleton::create(mName);
	setDefaultShape(Eigen::Vector3d(1.0, 0.0, 0.0));
	setCollision(false);
}

const dart::dynamics::SkeletonPtr&
Character2D::
getSkeleton()
{
	return mSkeleton;
}

void
Character2D::
setDefaultShape(const Eigen::Vector3d& color)
{
	Eigen::Isometry3d pb2jT;
	pb2jT.setIdentity();
	pb2jT.translation() += Eigen::Vector3d(0.0, 0.0, 0.0);
	SkelMaker::makeFree2DJointBody(mName + "_bodyJoint", mSkeleton, nullptr, 
		SHAPE_TYPE::BOX, Eigen::Vector3d(0.05, 0.05, 0.2), 
		pb2jT, Eigen::Isometry3d::Identity());

	pb2jT.setIdentity();
	pb2jT.translation() += Eigen::Vector3d(0.0, 0.0, 0.2);
	SkelMaker::makeWeldJointBody(mName + "_headJoint", mSkeleton, mSkeleton->getRootBodyNode(), 
		SHAPE_TYPE::BALL, Eigen::Vector3d::UnitX() * 0.08, 
		pb2jT, Eigen::Isometry3d::Identity());
}

void 
Character2D::
setVelocity(const Eigen::Vector3d& vel)
{
	mSkeleton->setVelocities(vel);
}

void
Character2D::
setCollision(bool enable)
{
	for(int i=0;i<mSkeleton->getNumBodyNodes();i++)
	{
		mSkeleton->getSkeleton()->getBodyNode(i)->setCollidable(enable);
	}
}