#undef NDEBUG
#include "Normalizer.h"
#include "../extern/ICA/plugin/MotionGenerator.h"
#include <iostream>
#include <fstream>

Normalizer::Normalizer(std::string xNormalPath, std::string yNormalPath)
{
	std::ifstream in;
	dimX = 14;
	dimY = 147;

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

/*Eigen::VectorXd 
Normalizer::
normalizeState(Eigen::VectorXd state)
{
	assert(state.rows() == dimY+12+1 + 3 + 8 +4);//+3 +2+8);
	// assert(state.rows() == dimY+12+1 + 3);
	Eigen::VectorXd normalizedState(state.rows());

	normalizedState.segment(0, dimY) = state.segment(0, dimY) - yMean;
	normalizedState.segment(0, dimY) = normalizedState.segment(0, dimY).cwiseProduct(yStd.cwiseInverse());


	// int ballOffset = MotionRepresentation::posOffset;

	normalizedState[dimY+0] = state[dimY+0] /(1400.0/sqrt(3.0));
	normalizedState[dimY+1] = state[dimY+1] /(200.0/sqrt(3.0));
	normalizedState[dimY+2] = state[dimY+2] /(750.0/sqrt(3.0));

	normalizedState[dimY+3] = state[dimY+3] /(1400.0/sqrt(3.0));
	normalizedState[dimY+4] = state[dimY+4] /(200.0/sqrt(3.0));
	normalizedState[dimY+5] = state[dimY+5] /(750.0/sqrt(3.0));

	// normalizedState.segment(dimY, 3) = normalizedState.segment(dimY, 3).cwiseProduct(yStd.segment(ballOffset, 3).cwiseInverse());


	normalizedState.segment(dimY+6,6) = state.segment(dimY+6, 6)/(1400.0/sqrt(3.0));
	normalizedState[dimY+12] = state[dimY+12]/30.0;

	normalizedState.segment(dimY+13,3) = state.segment(dimY+13,3)/100.0 * 4.0;

	normalizedState.segment(dimY+16,12) = state.segment(dimY+16,12);



	// normalizedState.segment(dimY+16,10) = state.segment(dimY+16,10);
	// normalizedState.segment(dimY+9,8) = state.segment(dimY+9,8);


	return normalizedState;
}
*/

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


// ad-hoc setting. the action type part is different from other part.
Eigen::VectorXd
Normalizer::
denormalizeAction(Eigen::VectorXd action)
{
	//critical action time dimension
	int offset = 1;
	// std::cout<<action.rows()<<std::endl;

	// -2 is for the hand contact which are not contained in xMean
	assert(action.rows()+offset-2 == dimX);

	// std::cout<<"------------------"<<std::endl;
	// std::cout<<action.transpose()<<std::endl;

	Eigen::VectorXd allignedAction(action.rows()-2);
	allignedAction.segment(4,5) = action.segment(0,5);
	allignedAction.segment(0,4) = action.segment(5,4);
	allignedAction.segment(9,4) = action.segment(9,4);



	Eigen::VectorXd denormalizedAction(action.rows()-2);

	// std::cout<<allignedAction.transpose()<<std::endl;
	// std::cout<<std::endl;

	// std::cout<<xMean.transpose()<<std::endl;
	// std::cout<<xStd.transpose()<<std::endl;
	// exit(0);
	denormalizedAction = allignedAction.cwiseProduct(xStd.segment(0,action.rows()-2));
	denormalizedAction = denormalizedAction + xMean.segment(0,action.rows()-2);

	denormalizedAction.segment(4,5) = allignedAction.segment(4,5);
	

	Eigen::VectorXd extendedAction(action.rows()+1);
	extendedAction.setZero();
	extendedAction.segment(0,action.rows()-2) = denormalizedAction;
	extendedAction.segment(action.rows()-2+offset,2) = action.segment(action.rows()-2,2);
	// exit(0);
	return extendedAction;
}


// Eigen::VectorXd
// Normalizer::
// normalizeAction(Eigen::VectorXd action)
// {
// 	assert(action.rows() == dimX);
// 	// std::cout<<"------------------"<<std::endl;
// 	// std::cout<<action.transpose()<<std::endl;

// 	// Eigen::VectorXd allignedAction(action.rows());
// 	// allignedAction.segment(4,8) = action.segment(0,8);
// 	// allignedAction.segment(0,4) = action.segment(8,4);
// 	// allignedAction.segment(12,dimX-12) = action.segment(12,dimX-12);

// 	Eigen::VectorXd normalizedAction(action.rows());

// 	// std::cout<<allignedAction.transpose()<<std::endl;
// 	// std::cout<<std::endl;

// 	// std::cout<<xMean.transpose()<<std::endl;
// 	// std::cout<<xStd.transpose()<<std::endl;

// 	// exit(0);
// 	normalizedAction = action - xMean;
// 	normalizedAction = action.cwiseProduct(xStd.cwiseInverse());

// 	// normalizedAction[18] = action[18];
// 	return normalizedAction;
// }

