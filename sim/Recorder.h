#ifndef __RECORDER_H__
#define __RECORDER_H__
#include "Environment.h"
class Recorder
{
public:
	Recorder();
	// std::vector<Eigen::Vector3d> obstaclePositions;
	// std::vector<Eigen::VectorXd> skelPositions;
	std::vector<Eigen::Vector6d> ballPositions;

	std::vector<std::vector<Eigen::Vector3d>> obstaclePositions;
	std::vector<std::vector<Eigen::VectorXd>> skelPositions;

	void recordCurrentFrame(Environment* env);
	void loadFrame(Environment* env, int frame);

	int loadLastFrame(Environment* env);

	int getNumFrames()
	{
		return skelPositions.size();
	}
};


#endif