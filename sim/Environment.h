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

// #define _ID_P 0
// #define _ID_V 2
// #define _ID_BALL_P 4
// #define _ID_BALL_V 6
// #define _ID_POSSESSION 8
// #define _ID_KICKABLE 9
// #define _ID_OTHERS 10
// #define _ID_DISTANCE_WALL 14
// #define _ID_GOALPOST_P 18

#define _ID_P 0
#define _ID_V 2
#define _ID_BALL_P 4
#define _ID_BALL_V 6
#define _ID_KICKABLE 8
#define _ID_GOALPOST_P 9


/*
	p.rows() + v.rows() + relativeBallP.rows() + relativeBallV.rows() +
	ballPossession.rows() + kickable.rows() + otherS.rows() + distanceWall.rows() + goalpostPositions.rows()
*/


// std::vector<std::string> skillSet{"ballChasing", "shooting", "ballBlocking"};

typedef std::pair<std::string, Eigen::Vector3d> GoalpostInfo;

class MapState
{
public:
	MapState(int numPrev){
		mNumPrev = numPrev;
		minimaps.reserve(numPrev);
		isFirst = true;
		updated = false;
	}
	void setCurState(Eigen::VectorXd curState);
	void endOfStep();

	void reset();
	std::vector<float> getVectorizedValue();


	int mNumPrev;
	std::vector<Eigen::VectorXd> minimaps;
	bool updated;
	bool isFirst;
};

class Environment
{
public:
	Environment(int control_Hz=30, int simulation_Hz=900, int numChars = 4);
	void initCharacters();
	void resetCharacterPositions();
	void initGoalposts();
	void initFloor();
	void initBall();
	void initPrevTargetPositions();

	void step();

	void stepAtOnce();

	void reset();
	bool isTerminalState();



	// For DeepRL
	// Eigen::VectorXd getState(int index);
	Eigen::VectorXd getState1(int index);
	Eigen::VectorXd getState(int index);
	Eigen::VectorXd getSchedulerState(int index);
	Eigen::VectorXd getLinearActorState(int index);

	// std::vector<Eigen::MatrixXd> getStateMinimap(int index);
	// Eigen::VectorXd getStateMinimap(int index);
	void setStateMinimap(int index);
	std::vector<double> getStateMinimapRGB(int index);

	double getReward(int index);
	double getLinearActorReward(int index);
	double getSchedulerReward(int index);
	std::vector<double> getRewards();

	Eigen::VectorXd getAction(int index){return mActions[index];}
	std::vector<Eigen::VectorXd> getActions(){return mActions;}

	void setAction(int index, const Eigen::VectorXd& a);
	void applyAction(int index);
	// void setActions(std::vector<Eigen::VectorXd> as);

	int getNumState(int index = 0){return getSchedulerState(index).size();}
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

	Eigen::VectorXd updateScoreBoard(std::string teamName = "");

	double addSkillReward(int index, int skillIndex);

	void initGoalState();

	void setLinearActorState(int index, Eigen::VectorXd linearActorState);


	Eigen::VectorXd normalizeNNState(Eigen::VectorXd state);
	Eigen::VectorXd unNormalizeNNState(Eigen::VectorXd outSubgoal);

	void updateState();

	void setHindsightGoal(Eigen::VectorXd randomSchedulerState);	
	Eigen::VectorXd getHindsightState(Eigen::VectorXd curState);
	double getHindsightReward(Eigen::VectorXd curHindsightState);

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

	int curDribblerIndex;

	bool mIsTerminalState;

	Eigen::VectorXd mKicked;
	Eigen::VectorXd mScoreBoard;
	Eigen::VectorXd mPrevScoreBoard;

	Eigen::VectorXd mAccScore;

	Eigen::VectorXd mTouch;

	int mNumIterations;

	std::vector<MapState*> mMapStates;

	std::vector<bool> goalRewardPaid;

	std::vector<Eigen::VectorXd> mStates;
	std::vector<Eigen::VectorXd> mPrevStates;
	std::vector<Eigen::VectorXd> mGoalStates;
	std::vector<Eigen::VectorXd> mWGoalStates;

	std::vector<Eigen::VectorXd> mSubgoalStates;
	std::vector<Eigen::VectorXd> mWSubgoalStates;

	std::vector<double> mSubGoalRewards;

	Eigen::VectorXd mHindsightGoalState;
	std::vector<Eigen::VectorXd> mPrevTargetPositions;
};

#endif