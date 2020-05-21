#ifndef __NORMALIZER_H__
#define __NORMALIZER_H__
#include <Eigen/Dense>

class Normalizer
{
	public:
	Normalizer(std::string xNormalPath, std::string yNormalPath);
	Eigen::VectorXd normalizeState(Eigen::VectorXd state);
	// Eigen::VectorXd denormalizeState(Eigen::VectorXd state);

	// Eigen::VectorXd normalizeAction();
	Eigen::VectorXd denormalizeAction(Eigen::VectorXd action);

	Eigen::VectorXd xMean, xStd;
	Eigen::VectorXd yMean, yStd;

	int dimX;
	int dimY;
};


#endif