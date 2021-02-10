#include "EnvironmentPython.h"
#include "../extern/ICA/plugin/utils.h"
#include <omp.h>
#include "dart/math/math.hpp"
#include "../vsports/common.h"
#include <iostream>
#include <GL/glut.h>

EnvironmentPython::
EnvironmentPython(int numAgent, std::string motion_nn_path, int numSlaves)
	:mNumSlaves(numSlaves)
{
	dart::math::seedRand();
	omp_set_num_threads(mNumSlaves);
	for(int i=0;i<mNumSlaves;i++)
	{
		mSlaves.push_back(new Environment(30, 180, numAgent, "../data/motions/basketData/motion/s_004_1_1.bvh", motion_nn_path));
	}
	initMotionGenerator(motion_nn_path);

	for(int i=0;i<mNumSlaves;i++)
	{
		if(i == 0)
			mSlaves[i]->initialize(mMotionGeneratorBatch, i, true);
		else
		{
			// mSlaves[i]->copyTutorialTrajectory(mSlaves[0]);
			mSlaves[i]->initialize(mMotionGeneratorBatch, i, false);
		}
	}
	mNumState = mSlaves[0]->getNumState();
	mNumAction = mSlaves[0]->getNumAction();
	slaveResets();

}
// For general properties

void
EnvironmentPython::
initMotionGenerator(std::string dataPath)
{
	mMotionGeneratorBatch = new ICA::dart::MotionGeneratorBatch(dataPath, mSlaves[0]->initDartNameIdMapping(), mNumSlaves);
}


void
EnvironmentPython::
resets()
{
	for(int id=0;id<mNumSlaves;++id)
		reset(id);
}


void
EnvironmentPython::
reset(int id)
{
	mSlaves[id]->reset();
}
void
EnvironmentPython::
slaveResets()
{
	for(int id=0;id<mNumSlaves;++id)
		slaveReset(id);
}


void
EnvironmentPython::
slaveReset(int id)
{
	mSlaves[id]->slaveReset();
}

void
EnvironmentPython::
foulReset(int id)
{
	mSlaves[id]->foulReset();
}
int
EnvironmentPython::
getNumState()
{
	return mNumState;
}

int
EnvironmentPython::
getNumAction()
{
	return mNumAction;
}

void
EnvironmentPython::
step(int id)
{
	mSlaves[id]->step();
}


bool
EnvironmentPython::
isTerminalState(int id)
{
	return mSlaves[id]->isTerminalState();
}

bool
EnvironmentPython::
isFoulState(int id)
{
	return mSlaves[id]->isFoulState();
}


np::ndarray
EnvironmentPython::
getState(int id, int index)
{
	return Wrapper::toNumPyArray(mSlaves[id]->getState(index));
}

np::ndarray
EnvironmentPython::
getCorrectActionType(int id, int index)
{
	exit(0);
	Eigen::VectorXd denormalizedAction = mSlaves[id]->mTutorialControlVectors[0][mSlaves[id]->curTrajectoryFrame];


	Eigen::VectorXd simpleActionType(2);

	int actionType = getActionTypeFromVec(denormalizedAction.segment(4,5));
	if(actionType == 2)
	{
		std::cout<<"Cur Action type is 2??"<<std::endl;
		std::cout<<mSlaves[id]->curTrajectoryFrame<<std::endl;
		exit(0);
	}
	actionType /= 3;
	simpleActionType.setZero();
	simpleActionType[actionType] = 1.0;


	return Wrapper::toNumPyArray(simpleActionType);
}

np::ndarray
EnvironmentPython::
getCorrectActionDetail(int id, int index)
{
	exit(0);
	Eigen::VectorXd denormalizedAction = mSlaves[id]->mTutorialControlVectors[0][mSlaves[id]->curTrajectoryFrame];

	Eigen::VectorXd normalizedAction(denormalizedAction.size()-5);
	normalizedAction.segment(0,4) = denormalizedAction.segment(0,4);
	normalizedAction.segment(4,5) = denormalizedAction.segment(9,5);

	normalizedAction = mSlaves[id]->mNormalizer->normalizeAction(normalizedAction);
	return Wrapper::toNumPyArray(normalizedAction);
}


void
EnvironmentPython::
setAction(np::ndarray np_array, int id, int index)
{

	bool reducedDim = false;

	Eigen::VectorXd action = Wrapper::toEigenVector(np_array);
	Eigen::VectorXd denormalizedAction;

	//** we change the dimension of action in denormalizeAction
	// ===Now we don't need to normalize action
	denormalizedAction = mSlaves[id]->mNormalizer->denormalizeAction(action);

	mSlaves[id]->setAction(index, denormalizedAction);
}


int
EnvironmentPython::
getActionTypeFromVec(Eigen::VectorXd action)
{
    int maxIndex = 0;
    double maxValue = -100;
    for(int i=0;i<action.size();i++)
    {
        if(action[i]> maxValue)
        {
        	maxValue= action[i];
        	maxIndex = i;
        }
    }

    return maxIndex;
}


int 
EnvironmentPython::
setActionType(int actionType, int id, int index, bool isNew)
{
	int constrainedActionType;
	int resetDuration = mSlaves[0]->resetDuration;
	if(mSlaves[id]->resetCount<=0)
		constrainedActionType = mSlaves[id]->setActionType(index, actionType, isNew);
	else
	{
		if(mSlaves[id]->randomPointTrajectoryStart)
		{
			Eigen::VectorXd actionTypeVector = mSlaves[id]->slaveResetTargetTrajectory[resetDuration-mSlaves[id]->resetCount].segment(0,5);
			constrainedActionType = mSlaves[id]->setActionType(index, getActionTypeFromVec(actionTypeVector), isNew);
		}
		else
		{
			Eigen::VectorXd actionTypeVector = mSlaves[id]->slaveResetTargetVector.segment(0,5);
			constrainedActionType = mSlaves[id]->setActionType(index, getActionTypeFromVec(actionTypeVector), isNew);
		}

	}


	return constrainedActionType;
}


double
EnvironmentPython::
getReward(int id, int index, int verbose)
{
	return mSlaves[id]->getReward(index, verbose);
}


void
EnvironmentPython::
stepsAtOnce()
{
	int num = getSimulationHz()/getControlHz();
// #pragma omp parallel for

	std::vector<std::vector<double>> concatControlVector;

	int resetDuration = mSlaves[0]->resetDuration;

	for(int id=0;id<mNumSlaves;++id)
	{
		if(mSlaves[id]->resetCount>resetDuration/2)
		{
			if(mSlaves[id]->randomPointTrajectoryStart)
			{
				mMotionGeneratorBatch->setBatchStateAndMotionGeneratorState(id, 
					mSlaves[id]->slaveResetPositionTrajectory[resetDuration - mSlaves[id]->resetCount], 
					mSlaves[id]->slaveResetBallPositionTrajectory[resetDuration - mSlaves[id]->resetCount]);
			}
			else
			{
				mMotionGeneratorBatch->setBatchStateAndMotionGeneratorState(id, mSlaves[id]->slaveResetPositionVector, mSlaves[id]->slaveResetBallPosition);
			}

		}
	}

	for(int id=0;id<mNumSlaves;++id)
	{
		if(mSlaves[id]->resetCount<=0)
			concatControlVector.push_back(eigenToStdVec(mSlaves[id]->getMGAction(0)));
		else
		{
			if(mSlaves[id]->randomPointTrajectoryStart)
			{
				concatControlVector.push_back(eigenToStdVec(mSlaves[id]->slaveResetTargetTrajectory[resetDuration-mSlaves[id]->resetCount]));
				Eigen::VectorXd actionTypeVector = mSlaves[id]->slaveResetTargetTrajectory[resetDuration-mSlaves[id]->resetCount].segment(0,5);
			}
			else
			{
				concatControlVector.push_back(eigenToStdVec(mSlaves[id]->slaveResetTargetVector));
				Eigen::VectorXd actionTypeVector = mSlaves[id]->slaveResetTargetVector.segment(0,5);
			}
		}
	}

	for(int id=0;id<mNumSlaves;++id)
		mSlaves[id]->saveEnvironment();
	// std::cout<<"concatControlVector.transpose() : "<<std::endl;
	// for(int i=0;i<concatControlVector.size();i++)
	// {
	// 	for(int j=0;j<concatControlVector[i].size();j++)
	// 	{
	// 		std::cout<<concatControlVector[i][j]<<" ";
	// 	}
	// 	std::cout<<std::endl;
	// 	std::cout<<"-----"<<std::endl;
		
	// }
	// std::cout<<std::endl;
	std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>>
	nextPoseAndContactsWithBatch = mMotionGeneratorBatch->generateNextPoseAndContactsWithBatch(concatControlVector);


	for(int id=0;id<mNumSlaves;++id)
	{
		mSlaves[id]->stepAtOnce(nextPoseAndContactsWithBatch[id]);
	}
}



void
EnvironmentPython::
endOfIteration()
{
	for(int i=0;i<mNumSlaves;i++)
	{
		mSlaves[i]->mNumIterations++;
	}
}

double
EnvironmentPython::
getNumIterations()
{
	return mSlaves[0]->mNumIterations;
}

void
EnvironmentPython::
setResetCount(int resetCount, int id)
{
	mSlaves[id]->resetCount = resetCount;
	mSlaves[id]->curFrame = mSlaves[id]->resetDuration - resetCount;
}

bool 
EnvironmentPython::
isOnFoulReset(int id)
{
	return mSlaves[id]->foulResetCount>0;
}

bool 
EnvironmentPython::
isOnResetProcess(int id)
{
	return mSlaves[id]->resetCount>=0;
}
int 
EnvironmentPython::
getResetDuration()
{
	return mSlaves[0]->resetDuration;
}

void
EnvironmentPython::
setToFoulState(int id)
{
	mSlaves[id]->mIsFoulState = true;
}
int 
EnvironmentPython::
getTypeFreq()
{
	return mSlaves[0]->typeFreq;
}


bool 
EnvironmentPython::
isTimeOut(int id)
{
	return mSlaves[id]->isTimeOut();
}

int
EnvironmentPython::
getSavedFrameDiff(int id)
{
	return mSlaves[id]->curFrame - mSlaves[id]->savedFrame;
}


using namespace boost::python;

BOOST_PYTHON_MODULE(pyvs)
{
	Py_Initialize();
	np::initialize();

	class_<EnvironmentPython>("Env", init<int, std::string, int>())
		.def("getNumState",&EnvironmentPython::getNumState)
		.def("getNumAction",&EnvironmentPython::getNumAction)
		.def("getSimulationHz",&EnvironmentPython::getSimulationHz)
		.def("getControlHz",&EnvironmentPython::getControlHz)
		.def("reset",&EnvironmentPython::reset)
		.def("resets",&EnvironmentPython::resets)
		.def("slaveReset",&EnvironmentPython::slaveReset)
		.def("foulReset",&EnvironmentPython::foulReset)
		.def("isTerminalState",&EnvironmentPython::isTerminalState)
		.def("isFoulState",&EnvironmentPython::isFoulState)
		.def("getState",&EnvironmentPython::getState)
		.def("getCorrectActionType",&EnvironmentPython::getCorrectActionType)
		.def("getCorrectActionDetail",&EnvironmentPython::getCorrectActionDetail)

		.def("setAction",&EnvironmentPython::setAction)
		.def("setActionType",&EnvironmentPython::setActionType)
		.def("getReward",&EnvironmentPython::getReward)

		.def("stepsAtOnce",&EnvironmentPython::stepsAtOnce)
		.def("isOnResetProcess",&EnvironmentPython::isOnResetProcess)
		.def("isOnFoulReset",&EnvironmentPython::isOnFoulReset)
		.def("setToFoulState",&EnvironmentPython::setToFoulState)

		.def("slaveResets",&EnvironmentPython::slaveResets)
		.def("getNumIterations",&EnvironmentPython::getNumIterations)

		.def("endOfIteration",&EnvironmentPython::endOfIteration)
		.def("setResetCount",&EnvironmentPython::setResetCount)
		.def("getResetDuration",&EnvironmentPython::getResetDuration)
		.def("getTypeFreq",&EnvironmentPython::getTypeFreq)
		.def("isTimeOut",&EnvironmentPython::isTimeOut)
		.def("getSavedFrameDiff",&EnvironmentPython::getSavedFrameDiff)
		;
}