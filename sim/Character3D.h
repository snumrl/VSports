#ifndef __VS_CHARACTER3D_H__
#define __VS_CHARACTER3D_H__
#include "dart/dart.hpp"

class Character3D
{
public:
	Character3D(const std::string& name);

	const dart::dynamics::SkeletonPtr& getSkeleton();
	// const dart::dynamics::SkeletonPtr& getArrow();

	std::string getName() {return mName;}
	std::string getTeamName() {return mName.substr(0,1);}

	dart::dynamics::SkeletonPtr mSkeleton;

	void copy(Character3D *c3d);

	// dart::dynamics::SkeletonPtr mArrow;
	std::string mName;

	double curLeftFingerPosition;
	double curRightFingerPosition;
	double curLeftFingerBallPosition;
	double curRightFingerBallPosition;

	bool blocked;

	int inputActionType;
	std::vector<int> availableActionTypes;

	Eigen::VectorXd prevSkelPositions;
	Eigen::VectorXd prevKeyJointPositions;

	Eigen::Isometry3d prevRootT;
	// double mDirection;
};

#endif