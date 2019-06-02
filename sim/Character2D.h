#ifndef __VS_CHARACTER2D_H__
#define __VS_CHARACTER2D_H__
#include "dart/dart.hpp"

class Character2D
{
public:
	Character2D(const std::string& name);

	const dart::dynamics::SkeletonPtr& getSkeleton();


	void setDefaultShape(const Eigen::Vector3d& color = Eigen::Vector3d(1.0, 0.0, 0.0));
	void setVelocity(const Eigen::Vector3d& vel);
	std::string getName() {return mName;}

protected:
	dart::dynamics::SkeletonPtr mSkeleton;
	std::string mName;
};

#endif