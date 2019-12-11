#ifndef __VS_ENVIRONMENT_H__
#define __VS_ENVIRONMENT_H__
#include "Character2D.h"
#include "BehaviorTree.h"

#define _ID_P 0
#define _ID_V 2
#define _ID_BALL_P 4
#define _ID_BALL_V 6
#define _ID_KICKABLE 8
#define _ID_GOALPOST_P 9
#define _ID_ALLY_P 17
#define _ID_ALLY_V 19
#define _ID_OP_DEF_P 21
#define _ID_OP_DEF_V 23
#define _ID_OP_ATK_P 25
#define _ID_OP_ATK_V 27
#define _ID_FACING_SIN 29
#define _ID_FACING_COS 30

// #define _ID_P 0
// #define _ID_V 2
// #define _ID_BALL_P 4
// #define _ID_BALL_V 6
// #define _ID_KICKABLE 8
// #define _ID_GOALPOST_P 9
// #define _ID_OP_DEF_P 17
// #define _ID_OP_DEF_V 19
// #define _ID_FACING_SIN 21
// #define _ID_FACING_COS 22

// // dummy for 1 agent
// #define _ID_ALLY_P 17
// #define _ID_ALLY_V 19
// #define _ID_OP_ATK_P 25
// #define _ID_OP_ATK_V 27

typedef std::pair<std::string, Eigen::Vector3d> GoalpostInfo;
class AgentEnvWindow;
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

	void stepAtOnce();

	void reset();
	bool isTerminalState();


	// For DeepRL
	Eigen::VectorXd getState(int index);
	Eigen::VectorXd getLocalState(int index);

	double getReward(int index);
	std::vector<double> getRewards();

	Eigen::VectorXd getAction(int index){return mActions[index];}
	std::vector<Eigen::VectorXd> getActions(){return mActions;}

	void setAction(int index, const Eigen::VectorXd& a);
	void applyAction(int index);

	int getNumState(int index = 0){return getState(index).size();}
	int getNumAction(int index = 0){return getAction(index).rows();}

	const dart::simulation::WorldPtr& getWorld(){return mWorld;}

	Character2D* getCharacter(int index){return mCharacters[index];}
	std::vector<Character2D*> getCharacters(){return mCharacters;}

	int getControlHz(){return mControlHz;}
	int getSimulationHz(){return mSimulationHz;}

	double getElapsedTime(){return mTimeElapsed;}

	std::vector<int> getCollidingWall(dart::dynamics::SkeletonPtr skel, double radius);

	void handleWallContact(dart::dynamics::SkeletonPtr skel, double radius, double me = 1.0);
	void handleBallContact(int index, double radius, double me = 0.5);
	void handlePlayerContact(int index1, int index2, double me = 0.5);
	void handlePlayerContacts(double me = 0.5);

	void boundBallVelocitiy(double maxVel);
	void dampBallVelocitiy(double dampPower);

	Eigen::VectorXd normalizeNNState(Eigen::VectorXd state);
	Eigen::VectorXd unNormalizeNNState(Eigen::VectorXd outSubgoal);

	void setVState(int index, Eigen::VectorXd latentState);

	void initBehaviorTree();
	Eigen::VectorXd getActionFromBTree(int index);

	void setHardcodedAction(int index);

	std::vector<int> getAgentViewImg(int index);

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
	std::vector<Eigen::VectorXd> mSimpleStates;
	std::vector<Eigen::VectorXd> mForces;

	double floorDepth = -0.1;

	bool mIsTerminalState;

	Eigen::VectorXd mScoreBoard;

	Eigen::VectorXd mAccScore;

	Eigen::VectorXd mTouch;

	int mNumIterations;

	std::vector<Eigen::VectorXd> mStates;
	std::vector<Eigen::VectorXd> mVStates;

	std::vector<BNode*> mBTs;
	double maxVel = 2.0;

	// AgentEnvWindow* mWindow;
};

#endif