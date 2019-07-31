#include "EnvironmentPython.h"
#include <omp.h>
#include "dart/math/math.hpp"
#include <iostream>

EnvironmentPython::
EnvironmentPython(int simulationHz)
	:mNumSlaves(16)
{
	dart::math::seedRand();
	omp_set_num_threads(mNumSlaves);
	for(int i=0;i<mNumSlaves;i++)
	{
		mSlaves.push_back(new Environment(30, simulationHz, 4));
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
	// std::cout<< toNumPyArray(mSlaves[id]->getState(index)).shape(0)<<std::endl;
	// exit(0);
	return toNumPyArray(mSlaves[id]->getState(index));
}

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
getReward(int id, int index)
{
	return mSlaves[id]->getReward(index);
}


void
EnvironmentPython::
stepsAtOnce()
{
	int num = getSimulationHz()/getControlHz();
#pragma omp parallel for
	for(int id=0;id<mNumSlaves;++id)
	{
		for(int j=0;j<num;j++)
			this->step(id);
	}
}

void
EnvironmentPython::
steps(int num)
{
#pragma omp parallel for
	for(int id=0;id<mNumSlaves;++id)
	{
		for(int j=0;j<num;j++)
			this->step(id);
	}
}

void
EnvironmentPython::
endOfIteration()
{
	for(int i=0;i<mNumSlaves;i++)
	{
		mSlaves[i]->mNumIterations++;
		if(mSlaves[i]->mNumIterations >= 400)
			mSlaves[i]->mNumIterations = 400;
	}
}

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
		.def("setAction",&EnvironmentPython::setAction)
		.def("getReward",&EnvironmentPython::getReward)
		.def("stepsAtOnce",&EnvironmentPython::stepsAtOnce)
		.def("steps",&EnvironmentPython::steps)
		.def("step",&EnvironmentPython::step)
		.def("resets",&EnvironmentPython::resets)
		.def("endOfIteration",&EnvironmentPython::endOfIteration);
}