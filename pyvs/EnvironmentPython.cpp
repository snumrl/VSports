#include "EnvironmentPython.h"
#include <omp.h>
#include "dart/math/math.hpp"
#include <iostream>
#include <GL/glut.h>

EnvironmentPython::
EnvironmentPython(int numAgent)
	:mNumSlaves(8)
{

	dart::math::seedRand();
	// omp_set_num_threads(mNumSlaves);
	for(int i=0;i<mNumSlaves;i++)
	{
		mSlaves.push_back(new Environment(30, 180, numAgent));
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

np::ndarray
EnvironmentPython::
getState(int id, int index)
{
	// std::cout<<"getState in wrapper"<<std::endl;
	// mSlaves[id]->getState(index);
	// std::cout<<"I got state"<<std::endl;
	// std::cout<< toNumPyArray(mSlaves[id]->getState(index)).shape(0)<<std::endl;
	// exit(0);
	return toNumPyArray(mSlaves[id]->getState(index));
}
np::ndarray
EnvironmentPython::
getLocalState(int id, int index)
{
	// std::cout<<"getState in wrapper"<<std::endl;
	// mSlaves[id]->getState(index);
	// std::cout<<"I got state"<<std::endl;
	// std::cout<< toNumPyArray(mSlaves[id]->getState(index)).shape(0)<<std::endl;
	// exit(0);
	return toNumPyArray(mSlaves[id]->getLocalState(index));
}
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
	mSlaves[id]->setAction(index, toEigenVector(np_array));
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

np::ndarray
EnvironmentPython::
getHardcodedAction(int id, int index)
{
	return toNumPyArray(mSlaves[id]->getActionFromBTree(index));
}

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

class GlutInitClass
{
public:
	GlutInitClass(){
		int argc = 1;
		char *argv[1] = {(char*)"Something"};
		glutInit(&argc, argv);
	}
};



using namespace boost::python;

BOOST_PYTHON_MODULE(pyvs)
{
	Py_Initialize();
	np::initialize();

	class_<EnvironmentPython>("Env", init<int>())
		.def("getNumState",&EnvironmentPython::getNumState)
		.def("getNumAction",&EnvironmentPython::getNumAction)
		.def("getSimulationHz",&EnvironmentPython::getSimulationHz)
		.def("getControlHz",&EnvironmentPython::getControlHz)
		.def("reset",&EnvironmentPython::reset)
		.def("isTerminalState",&EnvironmentPython::isTerminalState)
		.def("getState",&EnvironmentPython::getState)
		.def("getLocalState",&EnvironmentPython::getLocalState)
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
		.def("getHardcodedAction",&EnvironmentPython::getHardcodedAction)
		// .def("setLinearActorState",&EnvironmentPython::setLinearActorState)
		.def("endOfIteration",&EnvironmentPython::endOfIteration);
		// .def("setHindsightGoal",&EnvironmentPython::setHindsightGoal)
		// .def("getHindsightState",&EnvironmentPython::getHindsightState)
		// .def("getHindsightReward",&EnvironmentPython::getHindsightReward);

	class_<GlutInitClass>("GlutInitClass", init<>());

}