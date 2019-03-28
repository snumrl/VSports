#ifndef __VS_ENVIRONMENT_H__
#define __VS_ENVIRONMENT_H__
#include "Character2D.h"

class Environment
{
public:
	Environment(int control_Hz=30, int simulation_Hz=900, int numChars = 2);
	void step();

	void reset(bool random = true);
	bool isTerminalState();
	void initializeMotions();

	// For DeepRL
	Eigen::VectorXd getState(int index);
	std::vector<Eigen::VectorXd> getStates();

	double getReward(int index);
	std::vector<double> getRewards();

	Eigen::VectorXd getAction(int index){return mActions[index];}
	std::vector<Eigen::VectorXd> getActions(){return mActions;}

	void setAction(int index, const Eigen::VectorXd& a);
	void setActions(std::vector<Eigen::VectorXd> as);
	// void setGoal(int index, const Eigen::VectorXd& goal);
	// Eigen::VectorXd getGoal(int index);
	// Eigen::VectorXd getGoal(int index);

	int getNumState(int index = 0){return getState(index).rows();}
	int getNumAction(int index = 0){return getAction(index).rows();}

	const dart::simulation::WorldPtr& getWorld(){return mWorld;}

	Character2D* getCharacter(int index){return mCharacters[index];}
	std::vector<Character2D*> getCharacters(){return mCharacters;}

	int getControlHz(){return mControlHz;}
	int getSimulationHz(){return mSimulationHz;}

	double getElapsedTime(){return mTimeElapsed;}

public:
	dart::simulation::WorldPtr mWorld;
	int mNumChars;

	double mTimeElapsed;
	int mControlHz;
	int mSimulationHz;

	std::vector<Character2D*> mCharacters;
	std::vector<Eigen::VectorXd> mActions;





};

#endif