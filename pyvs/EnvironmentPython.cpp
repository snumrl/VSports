#include "EnvironmentPython.h"
#include <omp.h>
#include "dart/math/math.hpp"
#include <iostream>
#include <GL/glut.h>

EnvironmentPython::
EnvironmentPython(int numAgent, std::string motion_nn_path)
	:mNumSlaves(1)
{
	dart::math::seedRand();
	omp_set_num_threads(mNumSlaves);
	// std::cout<<""
	for(int i=0;i<mNumSlaves;i++)
	{
		mSlaves.push_back(new Environment(30, 180, numAgent, "../data/motions/basketData/motion/s_004_1_1.bvh", motion_nn_path));
	}
	mNumState = mSlaves[0]->getNumState();
	mNumAction = mSlaves[0]->getNumAction();

	// std::cout<<mNumState<<std::endl;
	// exit()
}
// For general properties
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
	// std::cout<<"Step in "<<id<<"'th slave"<<std::endl;
	mSlaves[id]->step();
}

void
EnvironmentPython::
stepAtOnce(int id)
{
	// std::cout<<"Step in "<<id<<"'th slave"<<std::endl;
	mSlaves[id]->stepAtOnce();
}

void
EnvironmentPython::
reset(int id)
{
	mSlaves[id]->reset();
}

void
EnvironmentPython::
resets()
{
	for(int id=0;id<mNumSlaves;++id)
		mSlaves[id]->reset();
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
	// std::cout<<"getState in wrapper"<<std::endl;
	// mSlaves[id]->getState(index);
	// std::cout<<"I got state"<<std::endl;
	// std::cout<< toNumPyArray(mSlaves[id]->getState(index)).shape(0)<<std::endl;
	// exit(0);
	// Eigen::VectorXd curState= mSlaves[id]->getState(index);
	// Eigen::VectorXd normalizedState = mNormalizer->normalizeState(curState);

	// std::cout<<"Original State : "<<std::endl;
	// std::cout<<curState.transpose()<<std::endl;
	// std::cout<<"Mean value :"<<std::endl;
	// std::cout<<mNormalizer->yMean.transpose()<<std::endl;

	// std::cout<<"std value :"<<std::endl;
	// std::cout<<mNormalizer->yStd.transpose()<<std::endl;
	// std::cout<<"Normalized State : "<<std::endl;
	// std::cout<<normalizedState.transpose()<<std::endl;

	return Wrapper::toNumPyArray(mSlaves[id]->getNormalizedState(index));
}

np::ndarray	
EnvironmentPython::
getHeightMapState(int id, int index)
{
	return Wrapper::toNumPyArray(mSlaves[id]->getHeightMapState(index));
}

void
EnvironmentPython::
goBackEnvironment(int id)
{
	mSlaves[id]->goBackEnvironment();
}
// np::ndarray
// EnvironmentPython::
// getLocalState(int id, int index)
// {
// 	// std::cout<<"getState in wrapper"<<std::endl;
// 	// mSlaves[id]->getState(index);
// 	// std::cout<<"I got state"<<std::endl;
// 	// std::cout<< toNumPyArray(mSlaves[id]->getState(index)).shape(0)<<std::endl;
// 	// exit(0);
// 	return Wrapper::toNumPyArray(mSlaves[id]->getLocalState(index));
// }


// np::ndarray
// EnvironmentPython::
// getSchedulerState(int id, int index)
// {
// 	// std::cout<<"getState in wrapper"<<std::endl;
// 	// mSlaves[id]->getState(index);
// 	// std::cout<<"I got state"<<std::endl;
// 	// std::cout<< toNumPyArray(mSlaves[id]->getState(index)).shape(0)<<std::endl;
// 	// exit(0);
// 	return toNumPyArray(mSlaves[id]->getSchedulerState(index));
// }
// np::ndarray
// EnvironmentPython::
// getLinearActorState(int id, int index)
// {
// 	// std::cout<<"getState in wrapper"<<std::endl;
// 	// mSlaves[id]->getState(index);
// 	// std::cout<<"I got state"<<std::endl;
// 	// std::cout<< toNumPyArray(mSlaves[id]->getState(index)).shape(0)<<std::endl;
// 	// exit(0);
// 	return toNumPyArray(mSlaves[id]->getLinearActorState(index));
// }

void
EnvironmentPython::
setAction(np::ndarray np_array, int id, int index)
{

	bool reducedDim = false;

	Eigen::VectorXd action = Wrapper::toEigenVector(np_array);
	Eigen::VectorXd denormalizedAction;
	Eigen::VectorXd ex_action(action.rows());
	if(reducedDim)
	{
		ex_action.setZero();
		ex_action.segment(0,4) = action;
		denormalizedAction = mSlaves[id]->mNormalizer->denormalizeAction(ex_action);
	}
	else
	{
		denormalizedAction = mSlaves[id]->mNormalizer->denormalizeAction(action);
	}


	// Eigen::VectorXd denormalizedAction = mSlaves[id]->mNormalizer->denormalizeAction(ex_action);

	// std::cout<<"Output Action :"<<std::endl;
	// std::cout<<action.transpose()<<std::endl;
	// std::cout<<"-----------------------------"<<std::endl;


	// std::cout<<"Denormalized Action :"<<std::endl;
	// std::cout<<denormalizedAction.transpose()<<std::endl;
	// std::cout<<"========================="<<std::endl;

	mSlaves[id]->setAction(index, denormalizedAction);
}


// void
// EnvironmentPython::
// setActions(np::ndarray np_array)
// {
// 	Eigen::MatrixXd action = toEigenMatrix(np_array);

// 	for(int id=0;id<mNumSlaves;++id)
// 	{
// 		mSlaves[id]->setAction(action.row(id).transpose());
// 	}
// }

double
EnvironmentPython::
getReward(int id, int index, int verbose)
{
	return mSlaves[id]->getReward(index, verbose);
}

// np::ndarray
// EnvironmentPython::
// getHardcodedAction(int id, int index)
// {
// 	return Wrapper::toNumPyArray(mSlaves[id]->getActionFromBTree(index));
// }

// double
// EnvironmentPython::
// getSchedulerReward(int id, int index)
// {
// 	return mSlaves[id]->getSchedulerReward(index);
// }


// double
// EnvironmentPython::
// getLinearActorReward(int id, int index)
// {
// 	return mSlaves[id]->getLinearActorReward(index);
// }

// void
// EnvironmentPython::
// setLinearActorState(int id, int index, np::ndarray np_array)
// {
// 	return mSlaves[id]->setLinearActorState(index, toEigenVector(np_array));
// }

void
EnvironmentPython::
stepsAtOnce()
{
	int num = getSimulationHz()/getControlHz();
// #pragma omp parallel for
	for(int id=0;id<mNumSlaves;++id)
	{
		// for(int j=0;j<num;j++)
		// 	this->step(id);
		// this->step
		this->stepAtOnce(id);
	}
}

// void
// EnvironmentPython::
// steps(int num)
// {
// #pragma omp parallel for
// 	for(int id=0;id<mNumSlaves;++id)
// 	{
// 		for(int j=0;j<num;j++)
// 			this->step(id);
// 	}
// }

void
EnvironmentPython::
endOfIteration()
{
	for(int i=0;i<mNumSlaves;i++)
	{
		mSlaves[i]->mNumIterations++;
	// std::cout<<mSlaves[i]->mNumIterations<<std::endl;
	}
}

double
EnvironmentPython::
getNumIterations()
{
	return mSlaves[0]->mNumIterations;
}

// void 
// EnvironmentPython::
// reconEnvFromState(int id, int index, np::ndarray curLocalState)
// {
// 	mSlaves[id]->reconEnvFromState(index, Wrapper::toEigenVector(curLocalState));
// }

// int
// EnvironmentPython::
// getNumBallTouch(int id)
// {
// 	return mSlaves[id]->getNumBallTouch();
// }

// void 
// EnvironmentPython::
// setHindsightGoal(np::ndarray randomSchedulerState)
// {
// 	mSlaves[0]->setHindsightGoal(toEigenVector(randomSchedulerState));
// }
// np::ndarray
// EnvironmentPython::
// getHindsightState(np::ndarray curState)
// {
// 	return toNumPyArray(mSlaves[0]->getHindsightState(toEigenVector(curState)));
// }

// double 
// EnvironmentPython::
// getHindsightReward(np::ndarray curHindsightState)
// {
// 	mSlaves[0]->getHindsightReward(toEigenVector(curHindsightState));
// }

// class GlutInitClass
// {
// public:
// 	GlutInitClass(){
// 		int argc = 1;
// 		char *argv[1] = {(char*)"Something"};
// 		glutInit(&argc, argv);
// 	}
// };



using namespace boost::python;

BOOST_PYTHON_MODULE(pyvs)
{
	Py_Initialize();
	np::initialize();

	class_<EnvironmentPython>("Env", init<int, std::string>())
		.def("getNumState",&EnvironmentPython::getNumState)
		.def("getNumAction",&EnvironmentPython::getNumAction)
		.def("getSimulationHz",&EnvironmentPython::getSimulationHz)
		.def("getControlHz",&EnvironmentPython::getControlHz)
		.def("reset",&EnvironmentPython::reset)
		.def("isTerminalState",&EnvironmentPython::isTerminalState)
		.def("isFoulState",&EnvironmentPython::isFoulState)
		.def("getState",&EnvironmentPython::getState)
		// .def("getLocalState",&EnvironmentPython::getLocalState)
		// .def("getSchedulerState",&EnvironmentPython::getSchedulerState)
		// .def("getLinearActorState",&EnvironmentPython::getLinearActorState)
		.def("setAction",&EnvironmentPython::setAction)
		.def("getReward",&EnvironmentPython::getReward)
		// .def("getSchedulerReward",&EnvironmentPython::getSchedulerReward)
		// .def("getLinearActorReward",&EnvironmentPython::getLinearActorReward)
		.def("stepsAtOnce",&EnvironmentPython::stepsAtOnce)
		.def("stepAtOnce",&EnvironmentPython::stepAtOnce)
		.def("step",&EnvironmentPython::step)
		.def("resets",&EnvironmentPython::resets)
		.def("getNumIterations",&EnvironmentPython::getNumIterations)
		.def("goBackEnvironment",&EnvironmentPython::goBackEnvironment)
		// .def("getHardcodedAction",&EnvironmentPython::getHardcodedAction)
		// .def("reconEnvFromState",&EnvironmentPython::reconEnvFromState)
		// .def("setLinearActorState",&EnvironmentPython::setLinearActorState)
		.def("endOfIteration",&EnvironmentPython::endOfIteration)
		// .def("getNumBallTouch",&EnvironmentPython::getNumBallTouch);
		// .def("setHindsightGoal",&EnvironmentPython::setHindsightGoal)
		// .def("getHindsightState",&EnvironmentPython::getHindsightState)
		// .def("getHindsightReward",&EnvironmentPython::getHindsightReward)
		;
}