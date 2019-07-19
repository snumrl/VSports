#ifndef __VS_ENVIRONMENT_H__
#define __VS_ENVIRONMENT_H__
#include "Character2D.h"

#define ID_P 0
#define ID_V 2
#define ID_BALL_P 4
#define ID_BALL_V 6
#define ID_POSSESSION 8
#define ID_KICKABLE 9
// #define ID_GOALPOST 10

typedef std::pair<std::string, Eigen::Vector3d> GoalpostInfo;

class Environment
{
public:
	Environment(int control_Hz=30, int simulation_Hz=900, int numChars = 4);
	void initCharacters();
	void resetCharacterPositions();
	void initGoalposts();
	void initFloor();
	void initBall();

	void step();

	void reset();
	bool isTerminalState();



	// For DeepRL
	// Eigen::VectorXd getState(int index);
	std::vector<double> getState(int index);

	// std::vector<Eigen::MatrixXd> getStateMinimap(int index);
	// Eigen::VectorXd getStateMinimap(int index);
	std::vector<double> getStateMinimap(int index);

	double getReward(int index);
	std::vector<double> getRewards();

	Eigen::VectorXd getAction(int index){return mActions[index];}
	std::vector<Eigen::VectorXd> getActions(){return mActions;}

	void setAction(int index, const Eigen::VectorXd& a);
	// void setActions(std::vector<Eigen::VectorXd> as);

	int getNumState(int index = 0){return getState(index).size();}
	int getNumAction(int index = 0){return getAction(index).rows();}

	const dart::simulation::WorldPtr& getWorld(){return mWorld;}

	Character2D* getCharacter(int index){return mCharacters[index];}
	std::vector<Character2D*> getCharacters(){return mCharacters;}

	int getControlHz(){return mControlHz;}
	int getSimulationHz(){return mSimulationHz;}

	double getElapsedTime(){return mTimeElapsed;}

	int getDribblerIndex();

	int getCollidingWall(dart::dynamics::SkeletonPtr skel, double radius);

	void handleWallContact(dart::dynamics::SkeletonPtr skel, double radius, double me = 1.0);
	void handleBallContact(int index, double radius, double me = 0.5);

	void boundBallVelocitiy(double maxVel);

	Eigen::VectorXd updateScoreBoard(std::string teamName = "");

public:
	dart::simulation::WorldPtr mWorld;
	int mNumChars;

	double mTimeElapsed;
	int mControlHz;
	int mSimulationHz;

	std::vector<Character2D*> mCharacters;

	dart::dynamics::SkeletonPtr floorSkel;
	dart::dynamics::SkeletonPtr ballSkel;
	dart::dynamics::SkeletonPtr wallSkel;

	std::vector<GoalpostInfo> mGoalposts;
	std::vector<Eigen::VectorXd> mActions;
	std::vector<Eigen::VectorXd> mStates;

	double floorDepth = -0.1;

	int curDribblerIndex;

	bool mIsTerminalState;

	Eigen::VectorXd mKicked;
	Eigen::VectorXd mScoreBoard;

	Eigen::VectorXd mAccScore;

	Eigen::VectorXd mTouch;

	int mNumIterations;
};

#endif