#ifndef __VS_EMVIRONMENT_PYTHON_H__
#define __VS_EMVIRONMENT_PYTHON_H__
#include "../sim/Environment.h"
#include <vector>
#include <string>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "../extern/ICA/plugin/MotionGenerator.h"
#include "../extern/ICA/plugin/MotionGeneratorBatch.h"
#include "WrapperFunctions.h"
#include "Normalizer.h"

class EnvironmentPython
{
public:
	EnvironmentPython(int numAgent, std::string motion_nn_path, int numSlaves);
	// For general properties
	int getNumState();
	int getNumAction();
	int getNumDofs();
	int getSimulationHz(){return mSlaves[0]->getSimulationHz();}
	int getControlHz(){return mSlaves[0]->getControlHz();}

	// For each slave
	void step(int id);
	void stepAtOnce(int id);
	// void steps(int id);
	void reset(int id);
	void resets();
	void slaveReset(int id);
	void foulReset(int id);
	void slaveResets();
	bool isTerminalState(int id);
	bool isFoulState(int id);
	np::ndarray getState(int id, int index);
	// np::ndarray getLocalState(int id, int index);
	void setAction(np::ndarray np_array, int id, int index);
	void setActions(np::ndarray np_array);
	double getReward(int id, int index, int verbose);
	double getNumIterations();
	int setActionType(int actionType, int id, int index, bool isNew = true);

	int isActionTypeChangingFrame(int id);
	// int getNumBallTouch(int id);

	void endOfIteration();

	// np::ndarray getHardcodedAction(int id, int index);

	// void reconEnvFromState(int id, int index, np::ndarray curLocalState);

	// For all slaves
	void stepsAtOnce();
	void initMotionGenerator(std::string dataPath);

	bool isOnResetProcess(int id);
	bool isOnFoulReset(int id);

	void setResetCount(int resetCount, int id);

    ICA::dart::MotionGeneratorBatch* mMotionGeneratorBatch;
	std::map<std::string, int> dartNameIdMap;



	int getActionTypeFromVec(Eigen::VectorXd action);

	np::ndarray getCorrectActionType(int id, int index);
	np::ndarray getCorrectActionDetail(int id, int index);

	int getResetDuration();
	// Normalizer* mNormalizer;
private:
	std::vector<Environment*> mSlaves;
	int mNumSlaves;
	int mNumState;
	int mNumAction;
};

#endif