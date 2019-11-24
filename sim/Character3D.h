// #ifndef __VS_CHARACTER3D_H__
// #define __VS_CHARACTER3D_H__
// #include "dart/dart.hpp"

// class Character3D
// {
// public:
// 	Character3D(const std::string& name);
// 	const dart::dynamics::SkeletonPtr& getSkeleton();

// 	void setDefaultShape(const Eigen::Vector3d& color = Eigen::Vector3d(1.0, 0.0, 0.0));
// 	void setVelocity(const Eigen::Vector2d& vel);
// 	std::string getName() {return mName;}
// 	std::string getTeamName() {return mName.substr(0,1);}

// 	void setCollision(bool enabl = true);
// 	void setDirection(Eigen::Vector2d direction) {mDirection = direction; mDirection.normalize();}
// 	Eigen::Vector2d getDirection() {return mDirection;}
// 	void directionStep(double timeStep = 1/600.0);
// 	void applyDirectionForce(double dforce);

// protected:
// 	dart::dynamics::SkeletonPtr mSkeleton;
// 	std::string mName;
// 	Eigen::Vector2d mDirection;
// 	double thetaDot;
// };

// #endif