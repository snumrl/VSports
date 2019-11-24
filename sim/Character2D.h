#ifndef __VS_CHARACTER2D_H__
#define __VS_CHARACTER2D_H__
#include "dart/dart.hpp"

class Character2D
{
public:
	Character2D(const std::string& name);

	const dart::dynamics::SkeletonPtr& getSkeleton();
	// const dart::dynamics::SkeletonPtr& getArrow();


	void setDefaultShape(const Eigen::Vector3d& color = Eigen::Vector3d(1.0, 0.0, 0.0));
	void setVelocity(const Eigen::Vector2d& vel);
	std::string getName() {return mName;}
	std::string getTeamName() {return mName.substr(0,1);}

	void setCollision(bool enabl = true);
	// void setDirection(double direction) {mSkeleton = direction;}
	// double getDirection() {return mDirection;}
	void directionStep(double timeStep = 1/600.0);
	void applyDirectionForce(double dforce);

protected:
	dart::dynamics::SkeletonPtr mSkeleton;
	// dart::dynamics::SkeletonPtr mArrow;
	std::string mName;
	// double mDirection;
};

#endif