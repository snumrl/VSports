#include "SkelHelper.h"
#include "SkelMaker.h"
using namespace dart;
using namespace dart::dynamics;

SkeletonPtr
SkelHelper::
makeFloor(double floorDepth)
{
	SkeletonPtr floor = Skeleton::create("floor");
	Eigen::Vector3d position = Eigen::Vector3d(0.0, 0.0, floorDepth);
	Eigen::Isometry3d cb2j;
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"floor", floor, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(8.0, 6.0, 0.01),
	Eigen::Isometry3d::Identity(), cb2j);
	return floor;
}

SkeletonPtr 
SkelHelper::
makeBall(double floorDepth)
{
	SkeletonPtr ball = Skeleton::create("ball");
	Eigen::Vector3d position = Eigen::Vector3d(0.0, 0.0, floorDepth + 0.08);
	Eigen::Isometry3d cb2j;
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeFree2DJointBody(
	"ball", ball, nullptr,
	SHAPE_TYPE::BALL,
	Eigen::Vector3d::UnitX()*0.08,
	Eigen::Isometry3d::Identity(), cb2j);
	return ball;
}

SkeletonPtr
SkelHelper::
makeWall(double floorDepth)
{
	double goalpostSize = 1.5;
	double goalLineSize = 6.0;
	SkeletonPtr wall = Skeleton::create("wall");
	Eigen::Vector3d position = Eigen::Vector3d(
		4.0, 
		goalpostSize/2.0 + (goalLineSize-goalpostSize)/4.0, 
		0.1 + floorDepth);
	Eigen::Isometry3d cb2j;
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall0_0", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(0.1, 2.25, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);


	position = Eigen::Vector3d(
		4.0, 
		-(goalpostSize/2.0 + (goalLineSize-goalpostSize)/4.0), 
		0.1 + floorDepth);
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall0_1", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(0.1, 2.25, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);

	position = Eigen::Vector3d(
		-4.0, 
		(goalpostSize/2.0 + (goalLineSize-goalpostSize)/4.0), 
		0.1 + floorDepth);
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall1_0", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(0.1, 2.25, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);

	position = Eigen::Vector3d(
		-4.0, 
		-(goalpostSize/2.0 + (goalLineSize-goalpostSize)/4.0), 
		0.1 + floorDepth);
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall1_1", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(0.1, 2.25, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);

	position = Eigen::Vector3d(0.0, 3.0, 0.1 + floorDepth);
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall2", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(8.0, 0.1, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);

	position = Eigen::Vector3d(0.0, -3.0, 0.1 + floorDepth);
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall3", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(8.0, 0.1, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);
	return wall;
}

SkeletonPtr 
SkelHelper::
makeGoalpost(Eigen::Vector3d position, std::string label)
{
	SkeletonPtr goalpost = Skeleton::create("goalpost_"+label);
	Eigen::Isometry3d cb2j;
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"goalpost_"+label, goalpost, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(0.1, 1.5, 0.5),
	Eigen::Isometry3d::Identity(), cb2j);
	return goalpost;
}
