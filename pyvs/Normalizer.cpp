#include "Normalizer.h"
#include "../extern/ICA/plugin/MotionGenerator.h"
#include <iostream>
#include <fstream>

Normalizer::Normalizer(std::string xNormalPath, std::string yNormalPath)
{
	std::ifstream in;
	dimX = 19;
	dimY = 146;

	in.open(xNormalPath);

	xMean.resize(dimX);
	xStd.resize(dimX);

	Basic::readNormal(in, xMean, xStd);

	in.close();



	in.open(yNormalPath);


	yMean.resize(dimY);
	yStd.resize(dimY);

	Basic::readNormal(in, yMean, yStd);

	in.close();
}

Eigen::VectorXd 
Normalizer::
normalizeState(Eigen::VectorXd state)
{
	assert(state.rows() == dimY+3);
	Eigen::VectorXd normalizedState(state.rows());
	normalizedState.segment(0, dimY) = state.segment(0, dimY) - yMean;
	normalizedState.segment(0, dimY) = state.segment(0, dimY) * yStd.cwiseInverse();

	int ballOffset = MotionRepresentation::posOffset;

	normalizedState.segment(dimY, 3) = state.segment(dimY, 3) - yMean.segment(ballOffset, 3);
	normalizedState.segment(dimY, 3) = state.segment(dimY, 3) * yStd.segment(ballOffset, 3).cwiseInverse();


	return normalizedState;
}


Eigen::VectorXd 
Normalizer::
denormalizeState(Eigen::VectorXd state)
{
	assert(state.rows() == dimY+3);
	Eigen::VectorXd denormalizedState(state.rows());
	denormalizedState.segment(0, dimY) = state.segment(0, dimY) + yMean;
	denormalizedState.segment(0, dimY) = state.segment(0, dimY) * yStd;

	int ballOffset = MotionRepresentation::posOffset;

	denormalizedState.segment(dimY, 3) = state.segment(dimY, 3) + yMean.segment(ballOffset, 3);
	denormalizedState.segment(dimY, 3) = state.segment(dimY, 3) * yStd.segment(ballOffset, 3);


	return denormalizedState;
}


Eigen::VectorXd 
Normalizer::
denormalizeAction(Eigen::VectorXd action)
{
	assert(action.rows() == dimX);
	Eigen::VectorXd denormalizedAction(action.rows());
	denormalizedAction = action + xMean;
	denormalizedAction = action * xStd;

	return denormalizedAction;
}

