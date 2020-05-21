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
	assert(state.rows() == dimY+9);
	Eigen::VectorXd normalizedState(state.rows());

	normalizedState.segment(0, dimY) = state.segment(0, dimY) - yMean;
	normalizedState.segment(0, dimY) = normalizedState.segment(0, dimY).cwiseProduct(yStd.cwiseInverse());


	// int ballOffset = MotionRepresentation::posOffset;

	normalizedState[dimY+0] = state[dimY+0] /(1400.0/sqrt(3.0));
	normalizedState[dimY+1] = state[dimY+1] /(200.0/sqrt(3.0));
	normalizedState[dimY+2] = state[dimY+2] /(750.0/sqrt(3.0));
	// normalizedState.segment(dimY, 3) = normalizedState.segment(dimY, 3).cwiseProduct(yStd.segment(ballOffset, 3).cwiseInverse());


	normalizedState.segment(dimY+3,6) = state.segment(dimY+3, 6)/(1400.0/sqrt(3.0));


	return normalizedState;
}


// Eigen::VectorXd 
// Normalizer::
// denormalizeState(Eigen::VectorXd state)
// {
// 	assert(state.rows() == dimY+9);
// 	Eigen::VectorXd denormalizedState(state.rows());
// 	denormalizedState.segment(0, dimY) = state.segment(0, dimY).cwiseProduct(yStd);
// 	denormalizedState.segment(0, dimY) = denormalizedState.segment(0, dimY) + yMean;

// 	// int ballOffset = MotionRepresentation::posOffset;


// 	denormalizedState[dimY+0] = state[dimY+0] *(1400.0/sqrt(3.0));
// 	denormalizedState[dimY+1] = state[dimY+1] *(200.0/sqrt(3.0));
// 	denormalizedState[dimY+2] = state[dimY+2] *(750.0/sqrt(3.0));

// 	// denormalizedState.segment(dimY, 3) = state.segment(dimY, 3).cwiseProduct(yStd.segment(ballOffset, 3));
// 	// denormalizedState.segment(dimY, 3) = denormalizedState.segment(dimY, 3) + yMean.segment(ballOffset, 3);

// 	denormalizedState.segment(dimY+3,6) = state.segment(dimY+3, 6)*(1400.0/sqrt(3.0));


// 	return denormalizedState;
// }


Eigen::VectorXd 
Normalizer::
denormalizeAction(Eigen::VectorXd action)
{
	assert(action.rows() == dimX);
	Eigen::VectorXd denormalizedAction(action.rows());
	// std::cout<<xMean.transpose()<<std::endl;
	// std::cout<<xStd.transpose()<<std::endl;
	// exit(0);
	denormalizedAction = action.cwiseProduct(xStd);
	denormalizedAction = denormalizedAction + xMean;


	return denormalizedAction;
}

