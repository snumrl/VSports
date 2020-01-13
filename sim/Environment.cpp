#include "Environment.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include "./BehaviorTree.h"
#include "dart/external/lodepng/lodepng.h"
#include "AgentEnvWindow.h"
#include <iostream>
#include <chrono>
#include <random>
#include <ctime>
#include <signal.h>

using namespace std;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::collision;
using namespace dart::constraint;

// p.rows() + v.rows() + ballP.rows() + ballV.rows() +
// 	ballPossession.rows() + kickable.rows() + goalpostPositions.rows()

Environment::
Environment(int control_Hz, int simulation_Hz, int numChars)
:mControlHz(control_Hz), mSimulationHz(simulation_Hz), mNumChars(numChars), mWorld(std::make_shared<dart::simulation::World>()),
mIsTerminalState(false), mTimeElapsed(0), mNumIterations(0), mSlowDuration(180), mNumBallTouch(0), endTime(15)
{
	srand((unsigned int)time(0));
	initBall();
	initCharacters();
	initGoalposts();
	initFloor();
	getNumState();

	mWorld->setTimeStep(1.0/mSimulationHz);

	// cout<<mWorld->getTimeStep()<<endl;
	// cout<<"11111"<<endl;
	// mWorld->getConstraintSolver()->removeAllConstraints();
	// cout<<"2222"<<endl;
	// mWindow = new AgentEnvWindow(0, this);
	// cout<<"3333"<<endl;
}

	// Create A team, B team players.
void
Environment::
initCharacters()
{
	if(mNumChars%2 != 0)
	{
		cout<<"The number of characters should be even number! Now "<<mNumChars<<endl;
		exit(0);
	}

	for(int i=0;i<mNumChars/2;i++)
	{
		mCharacters.push_back(new Character2D("A_" + to_string(i)));
	}
	for(int i=0;i<mNumChars/2;i++)
	{
		mCharacters.push_back(new Character2D("B_" + to_string(i)));
	}

	resetCharacterPositions();

	// Add skeletons
	for(int i=0;i<mNumChars;i++)
	{
		mWorld->addSkeleton(mCharacters[i]->getSkeleton());
	}

	for(int i=0;i<mNumChars;i++)
	{
		Eigen::VectorXd mAction;
		Eigen::VectorXd zeroVel(mCharacters[i]->getSkeleton()->getNumDofs());
		zeroVel.setZero();
		// Eigen::VectorXd controlBall(2);
		// Eigen::VectorXd mAction(zeroVel.size()+controlBall.size());
		// mAction << zeroVel, controlBall;

		Eigen::VectorXd touch(1);
		touch.setZero();
		mAccScore.resize(mNumChars);
		mAccScore.setZero();
		// mAction.resize(zeroVel.rows()+touch.rows());
		// mAction << zeroVel, touch;
		mAction.resize(3);
		mAction.setZero();
		mActions.push_back(mAction);
	}


	mStates.resize(mNumChars);
	mLocalStates.resize(mNumChars);
	mForces.resize(mNumChars);
	mBTs.resize(mNumChars);
	mFacingVels.resize(mNumChars);
	mKicked.resize(mNumChars);
	for(int i=0;i<mNumChars;i++)
	{
		mFacingVels[i] = 0.0;
	}
	initBehaviorTree();
}


void 
Environment::
resetCharacterPositions()
{
	if(mNumChars == 6)
	{
		std::vector<Eigen::Vector2d> charPositions;
		charPositions.push_back(Eigen::Vector2d(-2.0, 0.0));
		charPositions.push_back(Eigen::Vector2d(-1.0, 0.5));
		charPositions.push_back(Eigen::Vector2d(-1.0, -0.5));
		charPositions.push_back(Eigen::Vector2d(2.0, 0.0));
		charPositions.push_back(Eigen::Vector2d(1.0, 0.5));
		charPositions.push_back(Eigen::Vector2d(1.0, -0.5));

		for(int i=0;i<mNumChars;i++)
		{
			mCharacters[i]->getSkeleton()->setPositions(charPositions[i]);
			mCharacters[i]->getSkeleton()->setVelocities(Eigen::Vector2d(0.0, 0.0));
		}
	}
	else if(mNumChars == 4)
	{
		std::vector<Eigen::Vector2d> charPositions;
		charPositions.push_back(Eigen::Vector2d(1.0, 0.5));
		charPositions.push_back(Eigen::Vector2d(1.0, -0.5));
		charPositions.push_back(Eigen::Vector2d(2.0, 0.0));
		charPositions.push_back(Eigen::Vector2d(0.0, -0.0));

		for(int i=0;i<mNumChars;i++)
		{
			mCharacters[i]->getSkeleton()->setPositions(charPositions[i]);
			mCharacters[i]->getSkeleton()->setVelocities(Eigen::Vector2d(0.0, 0.0));
		}
		ballSkel->setPosition(0, 1.5);
	}
	else if(mNumChars == 2)
	{
		std::vector<Eigen::Vector2d> charPositions;
		charPositions.push_back(Eigen::Vector2d(-1.0, 0.0));
		charPositions.push_back(Eigen::Vector2d(1.0, 0.0));

		for(int i=0;i<mNumChars;i++)
		{
			mCharacters[i]->getSkeleton()->setPositions(charPositions[i]);
			mCharacters[i]->getSkeleton()->setVelocities(Eigen::Vector2d(0.0, 0.0));
		}
	}

	
}

void setSkelCollidable(SkeletonPtr skel, bool collidable = true)
{
	for(int i=0;i<skel->getNumBodyNodes();i++)
	{ 
		skel->getBodyNode(i)->setCollidable(collidable);
	}
}


void
Environment::
initGoalposts()
{
	GoalpostInfo teamA("A", Eigen::Vector3d(-4.0, 0.0, 0.25 + floorDepth));
	GoalpostInfo teamB("B", Eigen::Vector3d(4.0, 0.0, 0.25 + floorDepth));

	mGoalposts.push_back(teamA);
	mGoalposts.push_back(teamB);

	wallSkel = SkelHelper::makeWall(floorDepth);
	// setSkelCollidable(wallSkel, false);
	mWorld->addSkeleton(wallSkel);
}

void 
Environment::
initFloor()
{
	floorSkel = SkelHelper::makeFloor();
	mWorld->addSkeleton(floorSkel);
	// setSkelCollidable(floorSkel, false);
}

void 
Environment::
initBall()
{
	ballSkel = SkelHelper::makeBall();
	setSkelCollidable(ballSkel, false);
	mWorld->addSkeleton(ballSkel);
}

void
Environment::
handleWallContact(dart::dynamics::SkeletonPtr skel, double radius, double me)
{
	std::vector<int> collidingWalls = getCollidingWall(skel, radius);

	double groundWidth = 4.0;
	double groundHeight = 3.0;
	Eigen::VectorXd cur_pos;
	for(int i=0;i<collidingWalls.size();i++)
	{
		switch(collidingWalls[i])
		{
			case 0:
			if(me*skel->getVelocity(0)>0)
				skel->setVelocity(0, -me*skel->getVelocity(0));
			cur_pos = skel->getPositions();
			cur_pos[0] = (groundWidth-radius) - (cur_pos[0]-(groundWidth-radius));
			skel->setPositions(cur_pos);
			skel->setForce(0, 0);
			break;
			case 1:
			if(me*skel->getVelocity(0)<0)
				skel->setVelocity(0, -me*skel->getVelocity(0));
			cur_pos = skel->getPositions();
			cur_pos[0] = (0) - (cur_pos[0]-(0));
			skel->setPositions(cur_pos);
			skel->setForce(0, 0);
			break;
			case 2:
			if(me*skel->getVelocity(1)>0)
				skel->setVelocity(1, -me*skel->getVelocity(1));
			cur_pos = skel->getPositions();
			// cout<<cur_pos.transpose()<<endl;
			cur_pos[1] = (groundHeight-radius) - (cur_pos[1]-(groundHeight-radius));
			// cout<<"2 "<<cur_pos.transpose()<<endl;
			// cout<<endl;
			skel->setPositions(cur_pos);
			skel->setForce(1, 0);
			break;
			case 3:
			if(me*skel->getVelocity(1)<0)
				skel->setVelocity(1, -me*skel->getVelocity(1));
			cur_pos = skel->getPositions();
			// cout<<cur_pos.transpose()<<endl;
			cur_pos[1] = -(groundHeight-radius) - (cur_pos[1]+(groundHeight-radius));
			// cout<<"3 "<<cur_pos.transpose()<<endl;
			// cout<<endl;
			skel->setPositions(cur_pos);
			skel->setForce(1, 0);
			break;
			default: 
			break;
		}
	}
	
}

void
Environment::
handleBallContact(int index, double radius, double me)
{
	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	double ballDistance = (ballSkel->getPositions() - skel->getPositions().segment(0,2)).norm();

	double kickPower = mActions[index][2];

	// if(kickPower>=1)
	// 	kickPower = 1;
	if(kickPower > 1.0)
		kickPower = 1.0;

	if(kickPower > 0)
	{
		kickPower += 0.3;
		// kickPower = 0.5;
		mKicked[index] = mSlowDuration;
		mNumBallTouch+= 1;
		// kickPower = 1.0/(exp(-kickPower)+1);
		// cout<<"Kicked!"<<endl;
		// kickPower = 1.0;
		Eigen::VectorXd direction = skel->getVelocities().segment(0,2).normalized();
		ballSkel->setVelocities(direction*(1.0+me)*(1.0*kickPower));
		// ballSkel->setVelocities(skel->getVelocities().segment(0,2).normalized());
	}

}

void
Environment::
handlePlayerContact(int index1, int index2, double me)
{
	SkeletonPtr skel1 = mCharacters[index1]->getSkeleton();
	SkeletonPtr skel2 = mCharacters[index2]->getSkeleton();

	double playerRadius = 0.25;

	Eigen::Vector2d skel1Positions = skel1->getPositions().segment(0,2);
	Eigen::Vector2d skel2Positions = skel2->getPositions().segment(0,2);

	if((skel1Positions - skel2Positions).norm() < playerRadius)
	{
		Eigen::Vector2d skel1Velocities = skel1->getVelocities().segment(0,2);;
		Eigen::Vector2d skel2Velocities = skel2->getVelocities().segment(0,2);;

		Eigen::Vector2d p2pVector = skel1Positions - skel2Positions;

		p2pVector.normalize();

		// check the direction of velocities
		Eigen::Vector2d relativeVel = skel1Velocities - skel2Velocities;
		if(relativeVel.dot(p2pVector) >= 0)
			return;

		double relativeVel_p2p = p2pVector.dot(relativeVel);
		double skel1Vel_p2p = p2pVector.dot(skel1Velocities);
		double skel2Vel_p2p = p2pVector.dot(skel2Velocities);
		double meanVel_p2p = (skel1Vel_p2p+skel2Vel_p2p)/2.0;

		Eigen::Vector2d p2pNormalVector = Eigen::Vector2d(p2pVector[1], -p2pVector[0]); 

		Eigen::Vector2d skel1NewVelocities, skel2NewVelocities;
		skel1NewVelocities = (meanVel_p2p + (skel2Vel_p2p-meanVel_p2p) * me) * p2pVector + skel1Velocities.dot(p2pNormalVector) * p2pNormalVector;
		skel2NewVelocities = (meanVel_p2p + (skel1Vel_p2p-meanVel_p2p) * me) * p2pVector + skel2Velocities.dot(p2pNormalVector) * p2pNormalVector;

		Eigen::Vector2d skel1VelocitiesFull = skel1->getVelocities();
		Eigen::Vector2d skel2VelocitiesFull = skel1->getVelocities();
		skel1VelocitiesFull.segment(0,2) = skel1NewVelocities;
		skel2VelocitiesFull.segment(0,2) = skel2NewVelocities;

		skel1->setVelocities(skel1VelocitiesFull);
		skel2->setVelocities(skel2VelocitiesFull);
	}
}

void
Environment::
handlePlayerContacts(double me)
{
	for(int i=1;i<mCharacters.size();i++)
	{
		if(i==3)
			continue;
		for(int j=0;j<i;j++)
		{
			handlePlayerContact(i,j,me);
		}
	}
}

void 
Environment::
boundBallVelocitiy(double ballMaxVel)
{
	Eigen::VectorXd ballVel = ballSkel->getVelocities();
	// cout<<"ballVel size: "<<ballVel.size()<<endl;
	for(int i=0;i<ballVel.size();i++)
	{
		// cout<<"i:"<<i<<endl;
		if(abs(ballVel[i])>ballMaxVel)
		{
			ballVel[i] = ballVel[i]/abs(ballVel[i])*ballMaxVel;
		}
	}
	ballSkel->setVelocities(ballVel);	
}

void 
Environment::
dampBallVelocitiy(double dampPower)
{
	Eigen::VectorXd ballForce = ballSkel->getForces();
	ballForce -= 1.0*dampPower*ballSkel->getVelocities();
	ballSkel->setForces(ballForce);
}


void
Environment::
step()
{
	mTimeElapsed += 1.0 / (double)mSimulationHz;
	
	for(int i=0;i<mCharacters.size();i++)
	{
		applyAction(i);
	}
	// cout<<"11111111!"<<endl;

	handlePlayerContacts(0.7);

	for(int i=0;i<mCharacters.size();i++)
	{
		handleWallContact(mCharacters[i]->getSkeleton(), 0.08, 0.5);


		if(mStates[i][_ID_KICKABLE] == 1)
		{
			// cout<<"here right?"<<endl;
			// if(mKicked[i]<=0.5)
				handleBallContact(i, 0.12, 2.5);
		}
	}

	handleWallContact(ballSkel, 0.08, 0.8);

	boundBallVelocitiy(6.0);
	dampBallVelocitiy(0.8);
	// cout<<ballSkel->getVelocities().norm()<<endl;

	for(int i=0;i<mCharacters.size();i++)
	{
		mKicked[i] = max(0, mKicked[i]-1);
	}

	// cout<<mCharacters[0]->getSkeleton()->getForces().transpose()<<endl;

	mWorld->step();
}


void
Environment::
stepAtOnce()
{
	// cout<<"Start"<<endl;
	int sim_per_control = this->getSimulationHz()/this->getControlHz();
	for(int i=0;i<sim_per_control;i++)
	{
		this->step();
	}
	// getRewards();
	// cout<<"Here?"<<endl;
	// mWindow->display();
	// cout<<"end"<<endl;
}


Eigen::VectorXd
Environment::
getState(int index)
{
	// cout<<"getState "<<index<<" start"<<endl;
	// Character's state
	Eigen::Vector2d p,v;
	Character2D* character = mCharacters[index];
	p = character->getSkeleton()->getPositions().segment(0,2);
	v = character->getSkeleton()->getVelocities().segment(0,2);

	// Eigen::VectorXd distanceWall(4);

	// Ball's state
	Eigen::Vector2d ballP, ballV;
	ballP = ballSkel->getPositions();
	ballV = ballSkel->getVelocities();

	Eigen::Vector2d relativeBallP, relativeBallV;
	relativeBallP = ballP - p;
	relativeBallV = ballV - v;

	std::string teamName = character->getName().substr(0,1);
	// if(teamName == mGoalposts[0].first)
	// 	distanceWall << 4-ballP[0], 4+ballP[0], 3-ballP[1], 3+ballP[1];
	// else	
	// 	distanceWall << 4+ballP[0], 4-ballP[0], 3+ballP[1], 3-ballP[1];


	Eigen::VectorXd kickable(1);
	if(relativeBallP.norm()<0.25)
	{
		kickable[0] = 1;
	}
	else
	{
		kickable[0] = 0;
	}
	
	// Observation
	
	int numCurTeam=0;
	int numOppTeam=0;
	for(int i=0;i<mCharacters.size();i++)
	{
		if(teamName == mCharacters[i]->getTeamName())
			numCurTeam++;
		else
			numOppTeam++;
	}

	Eigen::VectorXd otherS((mCharacters.size() -1)*4);
	int count = 0;
	for(int i=0;i<mCharacters.size();i++)
	{
		if(mCharacters[i]->getName()!=character->getName())
		{
			if(mCharacters[i]->getTeamName()==character->getTeamName())
			{
				SkeletonPtr skel = mCharacters[i]->getSkeleton();
				otherS.segment(count*4,2) = skel->getPositions().segment(0,2) - p;
				otherS.segment(count*4+2,2) = skel->getVelocities().segment(0,2) - v;
				count++;
			}
		}
	}

	for(int i=0;i<mCharacters.size();i++)
	{
		if(mCharacters[i]->getName()!=character->getName())
		{
			if(mCharacters[i]->getTeamName()!=character->getTeamName())
			{
				SkeletonPtr skel = mCharacters[i]->getSkeleton();
				otherS.segment(count*4,2) = skel->getPositions().segment(0,2) - p;
				otherS.segment(count*4+2,2) = skel->getVelocities().segment(0,2) - v;
				count++;
			}
		}
	}



	double reviseStateByTeam = -1;
	if(teamName == mGoalposts[0].first)
	{
		reviseStateByTeam = 1;
	}


	Eigen::VectorXd simpleGoalpostPositions(8);
	if(teamName == mGoalposts[0].first)
	{
		simpleGoalpostPositions.segment(0,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - p;
		simpleGoalpostPositions.segment(2,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - p;
		simpleGoalpostPositions.segment(4,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - p;
		simpleGoalpostPositions.segment(6,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - p;

	}
	else
	{
		simpleGoalpostPositions.segment(0,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - p;
		simpleGoalpostPositions.segment(2,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - p;
		simpleGoalpostPositions.segment(4,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - p;
		simpleGoalpostPositions.segment(6,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - p;

	}

	// Eigen::VectorXd facingVel(1);
	// facingVel[0] = mFacingVels[index];

	Eigen::VectorXd slowed(1);
	slowed[0] = mKicked[index]/mSlowDuration;
	// cout<<facingVel[0]<<endl;

	// double facingAngle = character->getSkeleton()->getPosition(2);
	// facingAngle = M_PI * (1-reviseStateByTeam)/2.0 + facingAngle;
	// facing[0] = sin(facingAngle);
	// facing[1] = cos(facingAngle);


	// cout<<index<<" "<<facing.transpose()<<endl;
	Eigen::VectorXd state;

	state.resize(p.rows() + v.rows() + ballP.rows() + ballV.rows() + kickable.rows() + simpleGoalpostPositions.rows()
		+ otherS.rows()+slowed.rows());

	count = 0;
	for(int i=0;i<p.rows();i++)
	{
		state[count++] = reviseStateByTeam * p[i];
	}
	for(int i=0;i<v.rows();i++)
	{
		state[count++] = reviseStateByTeam * v[i];
	}
	for(int i=0;i<ballP.rows();i++)
	{
		state[count++] = reviseStateByTeam * (ballP[i] - p[i]);
	}
	for(int i=0;i<ballV.rows();i++)
	{
		state[count++] = reviseStateByTeam * (ballV[i] - v[i]);
	}

	for(int i=0;i<kickable.rows();i++)
	{
		state[count++] = kickable[i];
	}
	for(int i=0;i<simpleGoalpostPositions.rows();i++)
	{
		state[count++] = reviseStateByTeam * simpleGoalpostPositions[i];
	}

	for(int i=0;i<otherS.rows();i++)
	{
		state[count++] = reviseStateByTeam * otherS[i];
	}
	// for(int i=0;i<facingVel.rows();i++)
	// {
	// 	state[count++] = facingVel[i];
	// }
	for(int i=0;i<slowed.rows();i++)
	{
		state[count++] = slowed[i];
	}
	// cout<<"get State : "<<state[_ID_FACING_V]<<endl;
	mStates[index] = state;
	// cout<<"getState end"<<endl;
	return normalizeNNState(state);
}

Eigen::Vector2d
rotate2DVector(Eigen::Vector2d vec, double theta)
{	
	Eigen::Matrix2d rot;
	rot << cos(theta), -sin(theta), sin(theta), cos(theta);
	return rot * vec;
}

Eigen::VectorXd
Environment::
getLocalState(int index)
{
	getState(index);
	double reviseStateByTeam = -1;
	if(mCharacters[index]->getTeamName() == mGoalposts[0].first)
	{
		reviseStateByTeam = 1;
	}
	// double facingAngle = mCharacters[index]->getSkeleton()->getPosition(2);
	double facingAngle = 0.0;
	facingAngle = M_PI * (1-reviseStateByTeam)/2.0 + facingAngle;
	// cout<<facingAngle<<" ";
	// double facingAngle = atan2(mStates[index][_ID_FACING_SIN], mStates[index][_ID_FACING_COS]);
	Eigen::VectorXd localState = mStates[index];

	if(mNumChars == 6)
	{
		// localState.segment(_ID_V, 2) = rotate2DVector(localState.segment(_ID_V, 2), -facingAngle);
		// localState.segment(_ID_BALL_P, 2) = rotate2DVector(localState.segment(_ID_BALL_P, 2), -facingAngle);
		// localState.segment(_ID_BALL_V, 2) = rotate2DVector(localState.segment(_ID_BALL_V, 2), -facingAngle);
		// localState.segment(_ID_GOALPOST_P, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P, 2), -facingAngle);
		// localState.segment(_ID_GOALPOST_P+2, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+2, 2), -facingAngle);
		// localState.segment(_ID_GOALPOST_P+4, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+4, 2), -facingAngle);
		// localState.segment(_ID_GOALPOST_P+6, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+6, 2), -facingAngle);
		// localState.segment(_ID_ALLY1_P, 2) = rotate2DVector(localState.segment(_ID_ALLY1_P, 2), -facingAngle);
		// localState.segment(_ID_ALLY1_V, 2) = rotate2DVector(localState.segment(_ID_ALLY1_V, 2), -facingAngle);
		// localState.segment(_ID_ALLY2_P, 2) = rotate2DVector(localState.segment(_ID_ALLY2_P, 2), -facingAngle);
		// localState.segment(_ID_ALLY2_V, 2) = rotate2DVector(localState.segment(_ID_ALLY2_V, 2), -facingAngle);
		// localState.segment(_ID_OP_DEF_P, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_P, 2), -facingAngle);
		// localState.segment(_ID_OP_DEF_V, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_V, 2), -facingAngle);
		// localState.segment(_ID_OP_ATK1_P, 2) = rotate2DVector(localState.segment(_ID_OP_ATK1_P, 2), -facingAngle);
		// localState.segment(_ID_OP_ATK1_V, 2) = rotate2DVector(localState.segment(_ID_OP_ATK1_V, 2), -facingAngle);
		// localState.segment(_ID_OP_ATK2_P, 2) = rotate2DVector(localState.segment(_ID_OP_ATK2_P, 2), -facingAngle);
		// localState.segment(_ID_OP_ATK2_V, 2) = rotate2DVector(localState.segment(_ID_OP_ATK2_V, 2), -facingAngle);
	}

	else if(mNumChars == 4)
	{
		localState.segment(_ID_V, 2) = rotate2DVector(localState.segment(_ID_V, 2), -facingAngle);
		localState.segment(_ID_BALL_P, 2) = rotate2DVector(localState.segment(_ID_BALL_P, 2), -facingAngle);
		localState.segment(_ID_BALL_V, 2) = rotate2DVector(localState.segment(_ID_BALL_V, 2), -facingAngle);
		localState.segment(_ID_GOALPOST_P, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P, 2), -facingAngle);
		localState.segment(_ID_GOALPOST_P+2, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+2, 2), -facingAngle);
		localState.segment(_ID_GOALPOST_P+4, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+4, 2), -facingAngle);
		localState.segment(_ID_GOALPOST_P+6, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+6, 2), -facingAngle);
		localState.segment(_ID_ALLY_P, 2) = rotate2DVector(localState.segment(_ID_ALLY_P, 2), -facingAngle);
		localState.segment(_ID_ALLY_V, 2) = rotate2DVector(localState.segment(_ID_ALLY_V, 2), -facingAngle);
		localState.segment(_ID_OP_DEF_P, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_P, 2), -facingAngle);
		localState.segment(_ID_OP_DEF_V, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_V, 2), -facingAngle);
		localState.segment(_ID_OP_ATK_P, 2) = rotate2DVector(localState.segment(_ID_OP_ATK_P, 2), -facingAngle);
		localState.segment(_ID_OP_ATK_V, 2) = rotate2DVector(localState.segment(_ID_OP_ATK_V, 2), -facingAngle);
	}
	// else if(mNumChars == 3)
	// {
	// 	localState.segment(_ID_V, 2) = rotate2DVector(localState.segment(_ID_V, 2), -facingAngle);
	// 	localState.segment(_ID_BALL_P, 2) = rotate2DVector(localState.segment(_ID_BALL_P, 2), -facingAngle);
	// 	localState.segment(_ID_BALL_V, 2) = rotate2DVector(localState.segment(_ID_BALL_V, 2), -facingAngle);
	// 	localState.segment(_ID_GOALPOST_P, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P, 2), -facingAngle);
	// 	localState.segment(_ID_GOALPOST_P+2, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+2, 2), -facingAngle);
	// 	localState.segment(_ID_GOALPOST_P+4, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+4, 2), -facingAngle);
	// 	localState.segment(_ID_GOALPOST_P+6, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+6, 2), -facingAngle);
	// 	localState.segment(_ID_ALLY_P, 2) = rotate2DVector(localState.segment(_ID_ALLY_P, 2), -facingAngle);
	// 	localState.segment(_ID_ALLY_V, 2) = rotate2DVector(localState.segment(_ID_ALLY_V, 2), -facingAngle);
	// 	localState.segment(_ID_OP_DEF_P, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_P, 2), -facingAngle);
	// 	localState.segment(_ID_OP_DEF_V, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_V, 2), -facingAngle);
	// 	localState.segment(_ID_OP_ATK_P, 2) = rotate2DVector(localState.segment(_ID_OP_ATK_P, 2), -facingAngle);
	// 	localState.segment(_ID_OP_ATK_V, 2) = rotate2DVector(localState.segment(_ID_OP_ATK_V, 2), -facingAngle);
	// }

	else if(mNumChars == 2)
	{
		// localState.segment(_ID_V, 2) = rotate2DVector(localState.segment(_ID_V, 2), -facingAngle);
		// localState.segment(_ID_BALL_P, 2) = rotate2DVector(localState.segment(_ID_BALL_P, 2), -facingAngle);
		// localState.segment(_ID_BALL_V, 2) = rotate2DVector(localState.segment(_ID_BALL_V, 2), -facingAngle);
		// localState.segment(_ID_GOALPOST_P, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P, 2), -facingAngle);
		// localState.segment(_ID_GOALPOST_P+2, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+2, 2), -facingAngle);
		// localState.segment(_ID_GOALPOST_P+4, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+4, 2), -facingAngle);
		// localState.segment(_ID_GOALPOST_P+6, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+6, 2), -facingAngle);
		// localState.segment(_ID_OP_P, 2) = rotate2DVector(localState.segment(_ID_OP_P, 2), -facingAngle);
		// localState.segment(_ID_OP_V, 2) = rotate2DVector(localState.segment(_ID_OP_V, 2), -facingAngle);
	}
	// cout<<"get localState : "<<localState[_ID_FACING_V]<<endl;

	mLocalStates[index] = localState;
	// cout<<mLocalStates[index].transpose()<<endl;



	// Eigen::VectorXd genState = localStateToOriginState(mLocalStates[index], 2);
	// cout<<(genState - mStates[index]).transpose()<<endl;
	return normalizeNNState(localState);
}

double
Environment::
getReward(int index, bool verbose)
{
	double reward = 0.0;
	Eigen::VectorXd p = mStates[index].segment(_ID_P,2);
	Eigen::VectorXd ballP = p + mStates[index].segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = p + (mStates[index].segment(_ID_GOALPOST_P, 2) + mStates[index].segment(_ID_GOALPOST_P+2, 2))/2.0;


	Eigen::VectorXd v = mStates[index].segment(_ID_V, 2);
	Eigen::VectorXd ballV = v + mStates[index].segment(_ID_BALL_V, 2);

	Eigen::VectorXd ballToGoalpost = centerOfGoalpost - ballP;


	// reward = 0.1 * exp(-(p-ballP).norm());
	// reward = 0.1 * v.dot((ballP-p).normalized());

	// reward -= 3.0 * exp(-1/ballToGoalpost.norm());

	if(ballV.norm()>0 && ballToGoalpost.norm()>0)
	{
		// double cosTheta =  ballV.normalized().dot(ballToGoalpost.normalized());
		// reward += ballV.norm()* exp(-2.0 * acos(cosTheta));
	}
	// cout<<reward<<endl;
	// else
	// 	reward += 0;

	// double effort = pow(mActions[index].segment(0,2).norm(),2) + pow(mActions[index][2],2) + pow(mActions[index][3],2);


	if(index == 2)
	{
		reward = 1.0/30.0;
	}



	// goal Reward
	double ballRadius = 0.1;
	double goalpostSize = 1.5;
	Eigen::Vector2d ballPosition = ballSkel->getPositions();
	Eigen::Vector2d widthVector = Eigen::Vector2d::UnitX();
	Eigen::Vector2d heightVector = Eigen::Vector2d::UnitY();

	// if(index == 0)
	// 	cout<<p.transpose()<<endl;
	// cout<<0<<endl;
	if(widthVector.dot(ballPosition)+ballRadius >= mGoalposts[1].second.x())
	{
	// cout<<1<<endl;
		if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
		{
	// cout<<2<<endl;
				// cout<<index<<" ";
			if(verbose && index == 0)
			{
				std::cout<<"Red Team GOALL!!"<<std::endl;
			}
			// if(!goalRewardPaid[index])

			if(mCharacters[index]->getTeamName() == "A")
				reward += 1;
			else
				reward -= 1;
			// goalRewardPaid[index] = true;
			mIsTerminalState = true;
		}
	}
	else if(widthVector.dot(ballPosition)-ballRadius < mGoalposts[0].second.x())
	{
		if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
		{
			if(verbose && index == 0)
			{
				// cout<<index<<" ";
				std::cout<<"Blue Team GOALL!!"<<std::endl;
			}
			// if(!goalRewardPaid[index])
			if(mCharacters[index]->getTeamName() == "A")
				reward -= 1;
			else
				reward += 1;
			// goalRewardPaid[index] = true;
			mIsTerminalState = true;
		}
	}

	return reward;
}


Eigen::Matrix2d getRotationMatrix(double theta)
{
	Eigen::Matrix2d rotation2dMatrix;
	rotation2dMatrix << cos(theta), -sin(theta),
						sin(theta), cos(theta);
	return rotation2dMatrix;
}



// used only in test window
std::vector<double> 
Environment::
getRewards()
{
	std::vector<double> rewards;
	for(int i=0;i<mNumChars;i++)
	{
		// // getLinearActorReward(i);
		// if (i==0)
		// {
		// 	// rewards.push_back();
		// 	mAccScore[i] += getReward(i);
		// }
		mAccScore[i] += getReward(i);

	}

	return rewards;
}

void
Environment::
applyAction(int index)
{

	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	mForces[index].segment(0,2) -= 4.0*skel->getVelocities().segment(0,2);
	skel->setForces(mForces[index]);

	Eigen::VectorXd vel = skel->getVelocities();

	double curMaxvel = maxVel;
	if(mKicked[index]>0)
		curMaxvel = (1-0.8*sqrt(mKicked[index]/mSlowDuration))*maxVel;

	if (vel.segment(0,2).norm() > curMaxvel)
		vel.segment(0,2) = vel.segment(0,2)/vel.segment(0,2).norm() * curMaxvel;

	// vel[2] = 0.0;

	// double maxFacingVel = 10.0;
	// if(mFacingVels[index] > maxFacingVel)
	// 	mFacingVels[index] = maxFacingVel;
	// else if(mFacingVels[index] < -maxFacingVel)
	// 	mFacingVels[index] = -maxFacingVel;
	// skel->setPosition(2, skel->getPosition(2)+mFacingVels[index]/mSimulationHz);
	// cout<<mActions[index][2]<<endl;
	// mFacingVels[index] += 100.0*mActions[index][2]/mSimulationHz;
	// mFacingVels[index] -= 0.04*mFacingVels[index];

	skel->setVelocities(vel);
}

double
directionToTheta(Eigen::Vector2d direction)
{
	if(direction.norm() == 0)
		return 0;
	direction.normalize();
	return atan2(direction[1], direction[0]);
}


void 
Environment::
setAction(int index, const Eigen::VectorXd& a)
{
	bool isNanOccured = false;

	for(int i=0;i<a.size();i++)
	{
		if(std::isnan(a[i]))
		{
			isNanOccured = true;
		}
	}
	if(!isNanOccured)
		mActions[index] = a;
	else
		mActions[index].setZero();


	if(mActions[index].segment(0,2).norm()>1.0)
	{
		mActions[index].segment(0,2) /= mActions[index].segment(0,2).norm();
	}
	// if(mActions[index].segment(2,1).norm()>1.0)
	// {
	// 	mActions[index].segment(2,1) /= mActions[index].segment(2,1).norm();
	// }



	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	
	double reviseStateByTeam = -1;
	if( mCharacters[index]->getTeamName() == mGoalposts[0].first)
		reviseStateByTeam = 1;

	Eigen::VectorXd applyingForce = mActions[index].segment(0,2);


	// double curFacingAngle = skel->getPosition(2);
	double curFacingAngle = 0.0;
	// if(curFacingAngle > M_PI)
	// {
	// 	curFacingAngle -= 2*M_PI;
	// }
	// if(curFacingAngle < -M_PI)
	// {
	// 	curFacingAngle += 2*M_PI;
	// }
	// skel->setPosition(2, curFacingAngle);
	// cout<<curFacingAngle<<" "<<(getFacingAngleFromLocalState(mLocalStates[index])+M_PI*(1-reviseStateByTeam)/2.0)<<endl;

	/// Add pi if it is not the red team
	// curFacingAngle +=  M_PI*(1-reviseStateByTeam)/2.0;
	// cout<<skel->getPosition(2)<<endl;	

	// double facingAngle = curFacingAngle + atan2(mActions[index][2], mActions[index][3]);

	applyingForce.segment(0,2) = rotate2DVector(mActions[index].segment(0,2), curFacingAngle);
	// applyingForce[2] *= reviseStateByTeam/10.0;
	// applyingForce.segment(0,2) = rotate2DVector(mActions[index].segment(0,2), 0);

	// applyingForce[2] = 0.0;

	// skel->setPosition(2, facingAngle + M_PI*(1-reviseStateByTeam)/2.0 );
	// skel->setPosition(2, 0 + M_PI*(1-reviseStateByTeam)/2.0 );

	mForces[index] = 100.0*reviseStateByTeam*applyingForce;
	// cout<<mForces[index].transpose()<<endl;
	// skel->setPosition(2, directionToTheta(skel->getVelocities().segment(0,2)));
}


bool
Environment::
isTerminalState()
{
	if(mTimeElapsed > endTime)
	{
		// cout<<"Time overed"<<endl;
		mIsTerminalState = true;
	}

	return mIsTerminalState;
}

void
Environment::
reset()
{
	Eigen::VectorXd ballPosition = ballSkel->getPositions();
	ballPosition[0] = 2.0 * (rand()/(double)RAND_MAX );
	ballPosition[1] = 1.5 * (rand()/(double)RAND_MAX ) - 1.5/2.0;
	ballSkel->setPositions(ballPosition);
	Eigen::VectorXd ballVel = ballSkel->getVelocities();
	// ballVel[0] = 4.0 * (rand()/(double)RAND_MAX ) - 2.0;
	// ballVel[1] = 4.0 * (rand()/(double)RAND_MAX ) - 2.0;
	ballVel.setZero();
	ballSkel->setVelocities(ballVel);

	if(mNumChars >= 4)
	{
		for(int i=0;i<mNumChars;i++)
		{
			SkeletonPtr skel = mCharacters[i]->getSkeleton();
			Eigen::VectorXd skelPosition = skel->getPositions();
			if(i < mNumChars/2)
			{
				skelPosition[0] = 2.0 * (rand()/(double)RAND_MAX );
				skelPosition[1] = 2.5 * (rand()/(double)RAND_MAX ) - 2.5/2;
				// skelPosition[2] = 2.0 * M_PI * (rand()/(double)RAND_MAX );
				// cout<<skelPosition[0]<<endl;
			}
			else
			{
				skelPosition[0] = 2.0 * (rand()/(double)RAND_MAX ) + 2.0;
				skelPosition[1] = 2.5 * (rand()/(double)RAND_MAX ) - 2.5/2;
				// skelPosition[2] = 2.0 * M_PI * (rand()/(double)RAND_MAX );
				// cout<<skelPosition[0]<<endl;
			}
			skel->setPositions(skelPosition);
			if(i == 3)
				skel->setPositions(skelPosition.setZero());
			Eigen::VectorXd skelVel = skel->getVelocities();
			skelVel[0] = 3.0 * (rand()/(double)RAND_MAX ) - 1.5;
			skelVel[1] = 3.0 * (rand()/(double)RAND_MAX ) - 1.5;
			// skelVel[2] = 0.0;
			skel->setVelocities(skelVel);
			if(i == 3)
				skel->setVelocities(skelVel.setZero());
			// if(i == 0 && rand()%2 == 0)
			// {
			// 	ballSkel->setPositions(skelPosition);
			// 	ballSkel->setVelocities(skelVel);
			// }
		}


	}
	else if(mNumChars == 2)
	{
		for(int i=0;i<mNumChars;i++)
		{
			SkeletonPtr skel = mCharacters[i]->getSkeleton();
			Eigen::VectorXd skelPosition = skel->getPositions();
			if(i < 1)
			{
				skelPosition[0] = 6.0 * (rand()/(double)RAND_MAX ) - 6.0/2;
				skelPosition[1] = 4.0 * (rand()/(double)RAND_MAX ) - 4.0/2;
				// skelPosition[2] = 2.0 * M_PI * (rand()/(double)RAND_MAX );
				// cout<<skelPosition[0]<<endl;
			}
			else
			{
				skelPosition[0] = 6.0 * (rand()/(double)RAND_MAX ) - 6.0/2;
				skelPosition[1] = 4.0 * (rand()/(double)RAND_MAX ) - 4.0/2;
				// skelPosition[2] = 2.0 * M_PI * (rand()/(double)RAND_MAX );
				// cout<<skelPosition[0]<<endl;
			}

			skel->setPositions(skelPosition);
			Eigen::VectorXd skelVel = skel->getVelocities();
			skelVel[0] = 3.0 * (rand()/(double)RAND_MAX ) - 1.5;
			skelVel[1] = 3.0 * (rand()/(double)RAND_MAX ) - 1.5;
			// skelVel[2] = 0.0;
			skel->setVelocities(skelVel);


			Eigen::VectorXd ballPosition = ballSkel->getPositions();
			ballPosition[0] = 6.0 * (rand()/(double)RAND_MAX ) - 6.0/2;
			ballPosition[1] = 4.0 * (rand()/(double)RAND_MAX ) - 4.0/2;
			ballSkel->setPositions(ballPosition);



			// if(i == 0 && rand()%2 == 0)
			// {
			// 	ballSkel->setPositions(skelPosition);
			// 	ballSkel->setVelocities(skelVel);
			// }
		}
	}
	
	for(int i=0;i<mNumChars;i++)
	{
		mFacingVels[i] = 0.0;
	}


	mIsTerminalState = false;
	mTimeElapsed = 0;

	mAccScore.setZero();
	mNumBallTouch = 0;


	// resetCharacterPositions();
}



// 0: positive x / 1: negative x / 2 : positive y / 3 : negative y
std::vector<int>
Environment::
getCollidingWall(SkeletonPtr skel, double radius)
{
	std::vector<int> collidingWall;

	double groundWidth = 4.0;
	double groundHeight = 3.0;

	Eigen::Vector2d centerVector = skel->getPositions().segment(0,2);
	Eigen::Vector2d widthVector = Eigen::Vector2d::UnitX();
	Eigen::Vector2d heightVector = Eigen::Vector2d::UnitY();

	double ballRadius = 0.1;
	double goalpostSize = 1.5;

	Eigen::Vector2d ballPosition = ballSkel->getPositions();


	if(skel->getName() == "ball")
	{
		if(widthVector.dot(ballPosition)-ballRadius <= mGoalposts[0].second.x())
		{
			if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
			{
				return collidingWall;
			}
		}
		if(widthVector.dot(ballPosition)+ballRadius >= mGoalposts[1].second.x())
		{
			if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
			{
				return collidingWall;
			}
		}
	}
	if(centerVector.dot(widthVector) >= (groundWidth-radius))
		collidingWall.push_back(0);
	else if(centerVector.dot(widthVector) <= -(0))
		collidingWall.push_back(1);
	else if(centerVector.dot(heightVector) >= (groundHeight-radius))
		collidingWall.push_back(2);
	else if(centerVector.dot(heightVector) <= -(groundHeight-radius))
		collidingWall.push_back(3);
	

	return collidingWall;
}

Eigen::VectorXd
Environment::
normalizeNNState(Eigen::VectorXd state)
{
	Eigen::VectorXd normalizedState = state;
	int numState = normalizedState.size();

	normalizedState.segment(0, 8) = state.segment(0,8)/8.0;
	normalizedState.segment(9, numState-10) = state.segment(9, numState-10)/8.0;
	// cout<<"Normalie NN state : "<<normalizedState[_ID_FACING_V]<<endl;
	return normalizedState;
}


Eigen::VectorXd
Environment::
unNormalizeNNState(Eigen::VectorXd outSubgoal)
{
	Eigen::VectorXd scaledSubgoal = outSubgoal;
	int numState = scaledSubgoal.size();
	scaledSubgoal.segment(0, 8) = outSubgoal.segment(0,8)*8.0;
	scaledSubgoal.segment(9, numState-10) = outSubgoal.segment(9, numState-10)*8.0;
	return scaledSubgoal;
}

Eigen::VectorXd softMax(Eigen::VectorXd input)
{
	Eigen::VectorXd softMaxVec(input.size());
	double total = 0;
	for(int i=0;i<input.size();i++)
	{
		softMaxVec[i] = exp(input[i]);
		total +=softMaxVec[i];
	}
	softMaxVec /= total;

	return softMaxVec;
}

void
Environment::
setVState(int index, Eigen::VectorXd latentState)
{
	mVStates[index] = latentState;
}

bool isNotCloseToBall(Eigen::VectorXd curState)
{
	return curState.segment(_ID_BALL_P,2).norm() >= 0.15;
}

bool isNotCloseToBall_soft(Eigen::VectorXd curState)
{
	return curState.segment(_ID_BALL_P,2).norm() >= 0.25;
}


// Eigen::VectorXd
// genActionNoise(int numRows)
// {
// 	Eigen::VectorXd actionNoise(numRows);
// 	// std::default_random_engine generator;
// 	std::random_device rd{};
//     std::mt19937 gen{rd()};
// 	std::normal_distribution<double> distribution1(0.0, 0.0);
// 	for(int i=0;i<2;i++)
// 	{
// 		actionNoise[i] = distribution1(gen);
// 	}

// 	std::normal_distribution<double> distribution2(0.0, 0.0);
// 	for(int i=2;i<4;i++)
// 	{
// 		actionNoise[i] = distribution2(gen);
// 	}
	
// 	std::normal_distribution<double> distribution3(0.0, 0.0);
// 	for(int i=4;i<5;i++)
// 	{
// 		actionNoise[i] = distribution3(gen);
// 	}
// 	actionNoise.setZero();
// 	// cout<<actionNoise.transpose()<<endl;
// 	return actionNoise;
// }

Eigen::VectorXd localStateToOriginState(Eigen::VectorXd localState, int mNumChars)
{
	// Eigen::VectorXd originState(localState.rows());
	Eigen::VectorXd originState = localState;
	double facingAngle = getFacingAngleFromLocalState(localState);

	if(mNumChars == 6)
	{
		originState.segment(_ID_V, 2) = rotate2DVector(localState.segment(_ID_V, 2), facingAngle);
		originState.segment(_ID_BALL_P, 2) = rotate2DVector(localState.segment(_ID_BALL_P, 2), facingAngle);
		originState.segment(_ID_BALL_V, 2) = rotate2DVector(localState.segment(_ID_BALL_V, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P+2, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+2, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P+4, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+4, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P+6, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+6, 2), facingAngle);
		originState.segment(_ID_ALLY1_P, 2) = rotate2DVector(localState.segment(_ID_ALLY1_P, 2), facingAngle);
		originState.segment(_ID_ALLY1_V, 2) = rotate2DVector(localState.segment(_ID_ALLY1_V, 2), facingAngle);
		originState.segment(_ID_ALLY2_P, 2) = rotate2DVector(localState.segment(_ID_ALLY2_P, 2), facingAngle);
		originState.segment(_ID_ALLY2_V, 2) = rotate2DVector(localState.segment(_ID_ALLY2_V, 2), facingAngle);
		originState.segment(_ID_OP_DEF_P, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_P, 2), facingAngle);
		originState.segment(_ID_OP_DEF_V, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_V, 2), facingAngle);
		originState.segment(_ID_OP_ATK1_P, 2) = rotate2DVector(localState.segment(_ID_OP_ATK1_P, 2), facingAngle);
		originState.segment(_ID_OP_ATK1_V, 2) = rotate2DVector(localState.segment(_ID_OP_ATK1_V, 2), facingAngle);
		originState.segment(_ID_OP_ATK2_P, 2) = rotate2DVector(localState.segment(_ID_OP_ATK2_P, 2), facingAngle);
		originState.segment(_ID_OP_ATK2_V, 2) = rotate2DVector(localState.segment(_ID_OP_ATK2_V, 2), facingAngle);


	}
	// if(mNumChars == 4)
	// {
	// 	originState.segment(_ID_V, 2) = rotate2DVector(localState.segment(_ID_V, 2), facingAngle);
	// 	originState.segment(_ID_BALL_P, 2) = rotate2DVector(localState.segment(_ID_BALL_P, 2), facingAngle);
	// 	originState.segment(_ID_BALL_V, 2) = rotate2DVector(localState.segment(_ID_BALL_V, 2), facingAngle);
	// 	originState.segment(_ID_GOALPOST_P, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P, 2), facingAngle);
	// 	originState.segment(_ID_GOALPOST_P+2, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+2, 2), facingAngle);
	// 	originState.segment(_ID_GOALPOST_P+4, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+4, 2), facingAngle);
	// 	originState.segment(_ID_GOALPOST_P+6, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+6, 2), facingAngle);
	// 	originState.segment(_ID_ALLY1_P, 2) = rotate2DVector(localState.segment(_ID_ALLY1_P, 2), facingAngle);
	// 	originState.segment(_ID_ALLY1_V, 2) = rotate2DVector(localState.segment(_ID_ALLY1_V, 2), facingAngle);
	// 	originState.segment(_ID_ALLY2_P, 2) = rotate2DVector(localState.segment(_ID_ALLY2_P, 2), facingAngle);
	// 	originState.segment(_ID_ALLY2_V, 2) = rotate2DVector(localState.segment(_ID_ALLY2_V, 2), facingAngle);
	// 	originState.segment(_ID_OP_DEF_P, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_P, 2), facingAngle);
	// 	originState.segment(_ID_OP_DEF_V, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_V, 2), facingAngle);
	// 	originState.segment(_ID_OP_ATK1_P, 2) = rotate2DVector(localState.segment(_ID_OP_ATK1_P, 2), facingAngle);
	// 	originState.segment(_ID_OP_ATK1_V, 2) = rotate2DVector(localState.segment(_ID_OP_ATK1_V, 2), facingAngle);
	// 	originState.segment(_ID_OP_ATK2_P, 2) = rotate2DVector(localState.segment(_ID_OP_ATK2_P, 2), facingAngle);
	// 	originState.segment(_ID_OP_ATK2_V, 2) = rotate2DVector(localState.segment(_ID_OP_ATK2_V, 2), facingAngle);


	// }



	else if(mNumChars == 4)
	{
		originState.segment(_ID_V, 2) = rotate2DVector(localState.segment(_ID_V, 2), facingAngle);
		originState.segment(_ID_BALL_P, 2) = rotate2DVector(localState.segment(_ID_BALL_P, 2), facingAngle);
		originState.segment(_ID_BALL_V, 2) = rotate2DVector(localState.segment(_ID_BALL_V, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P+2, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+2, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P+4, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+4, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P+6, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+6, 2), facingAngle);
		// cout<<(originState.segment(_ID_GOALPOST_P, 2) + originState.segment(_ID_P, 2)).transpose()<<endl;
		originState.segment(_ID_ALLY_P, 2) = rotate2DVector(localState.segment(_ID_ALLY_P, 2), facingAngle);
		originState.segment(_ID_ALLY_V, 2) = rotate2DVector(localState.segment(_ID_ALLY_V, 2), facingAngle);
		originState.segment(_ID_OP_DEF_P, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_P, 2), facingAngle);
		originState.segment(_ID_OP_DEF_V, 2) = rotate2DVector(localState.segment(_ID_OP_DEF_V, 2), facingAngle);
		originState.segment(_ID_OP_ATK_P, 2) = rotate2DVector(localState.segment(_ID_OP_ATK_P, 2), facingAngle);
		originState.segment(_ID_OP_ATK_V, 2) = rotate2DVector(localState.segment(_ID_OP_ATK_V, 2), facingAngle);

	}
	else if(mNumChars == 2)
	{
		originState.segment(_ID_V, 2) = rotate2DVector(localState.segment(_ID_V, 2), facingAngle);
		originState.segment(_ID_BALL_P, 2) = rotate2DVector(localState.segment(_ID_BALL_P, 2), facingAngle);
		originState.segment(_ID_BALL_V, 2) = rotate2DVector(localState.segment(_ID_BALL_V, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P+2, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+2, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P+4, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+4, 2), facingAngle);
		originState.segment(_ID_GOALPOST_P+6, 2) = rotate2DVector(localState.segment(_ID_GOALPOST_P+6, 2), facingAngle);
		originState.segment(_ID_OP_P, 2) = rotate2DVector(localState.segment(_ID_OP_P, 2), facingAngle);
		originState.segment(_ID_OP_V, 2) = rotate2DVector(localState.segment(_ID_OP_V, 2), facingAngle);
	}

	return originState;
}

double getFacingAngleFromLocalState(Eigen::VectorXd localState)
{
	Eigen::Vector2d goalToGoal = (localState.segment(_ID_GOALPOST_P, 2) - localState.segment(_ID_GOALPOST_P+6, 2));

	goalToGoal.normalize();
	return -atan2(goalToGoal[1], goalToGoal[0]);
}

double hardcodedShootingPower = 0.7;
double controlHz = 30.0;
double torqueScale = 0.4;

Eigen::VectorXd
actionCloseToBall(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd curVel = curState.segment(_ID_V,2);
	Eigen::VectorXd targetVel = curState.segment(_ID_BALL_P,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;

	// Eigen::Vector2d direction = targetVel - curVel;
	// direction.normalize();
	// action[2] = directionToTheta(direction);

	double curFacingAngle = getFacingAngleFromLocalState(localState);

	double targetFacingAngle = directionToTheta(curState.segment(_ID_V,2));
	double direction = targetFacingAngle - (curFacingAngle);
	// cout<<targetFacingAngle<<" "<<curFacingAngle<<" "<<curState[_ID_FACING_V]<<" "<<controlHz<<endl;
	// cout<<targetFacingAngle<<" "<<(curFacingAngle + curState[_ID_FACING_V]/controlHz)<<endl;
	if(direction > M_PI)
	{
		direction = direction - 2*M_PI;
	}
	if(direction < -M_PI)
	{
		direction = direction + 2*M_PI;
	}

	action[2] = torqueScale*direction;
	// cout<<action[2]<<endl;
	// action[2] = sin(facingAngle - curFacingAngle);
	// action[3] = cos(facingAngle - curFacingAngle);

	action.segment(0,2) = rotate2DVector(action.segment(0,2), -curFacingAngle);
	// action.segment(0,2) = rotate2DVector(action.segment(0,2), -curFacingAngle);
	action.segment(0,2).normalize();
	action[3] = -hardcodedShootingPower;
	
	return action;
}

Eigen::VectorXd
actionCloseToBallWithShooting(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd curVel = curState.segment(_ID_V,2);
	Eigen::VectorXd targetVel = curState.segment(_ID_BALL_P,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;

	double curFacingAngle = getFacingAngleFromLocalState(localState);

	double targetFacingAngle = directionToTheta(curState.segment(_ID_V,2));
	double direction = targetFacingAngle - (curFacingAngle);
	if(direction > M_PI)
		direction = direction - 2*M_PI;
	else if(direction < -M_PI)
		direction = direction + 2*M_PI;

	action[2] = torqueScale*direction;
	action.segment(0,2) = rotate2DVector(action.segment(0,2),  -(curFacingAngle));
	action.segment(0,2).normalize();

	action[3] = hardcodedShootingPower;

	
	return action;
}
bool isNotVelHeadingGoal(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd p = curState.segment(_ID_P, 2);
	Eigen::VectorXd v = curState.segment(_ID_V, 2);
	Eigen::VectorXd agentToGoalpost = (curState.segment(_ID_GOALPOST_P, 2) + curState.segment(_ID_GOALPOST_P+2, 2))/2.0;

	double cosTheta =  v.normalized().dot(agentToGoalpost.normalized());
	
	return cosTheta < 0.95 && curState.segment(_ID_BALL_P,2).norm() < 0.23;
}
Eigen::VectorXd
actionVelHeadingGoal(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd targetVel = (curState.segment(_ID_GOALPOST_P, 2) + curState.segment(_ID_GOALPOST_P+2, 2))/2.0;

	Eigen::VectorXd curVel = curState.segment(_ID_V,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel- curVel;

	double curFacingAngle = getFacingAngleFromLocalState(localState);

	double targetFacingAngle = directionToTheta(curState.segment(_ID_V,2));
	double direction = targetFacingAngle - (curFacingAngle);
	if(direction > M_PI)
		direction = direction - 2*M_PI;
	else if(direction < -M_PI)
		direction = direction + 2*M_PI;

	action[2] = torqueScale*direction;

	action.segment(0,2) = rotate2DVector(action.segment(0,2), -(curFacingAngle));
	action.segment(0,2).normalize();

	action[3] = -hardcodedShootingPower;
	
	return action;
}

bool isNotBallHeadingGoal(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = p + (curState.segment(_ID_GOALPOST_P, 2) + curState.segment(_ID_GOALPOST_P+2, 2))/2.0;


	Eigen::VectorXd v =curState.segment(_ID_V, 2);
	Eigen::VectorXd ballV = v + curState.segment(_ID_BALL_V, 2);

	Eigen::VectorXd ballToGoalpost = centerOfGoalpost - ballP;

	double cosTheta =  ballV.normalized().dot(ballToGoalpost.normalized());
	
	return cosTheta < 0.9 && curState.segment(_ID_BALL_P,2).norm() < 0.23;
}
Eigen::VectorXd
actionBallHeadingGoal(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd targetVel = (curState.segment(_ID_GOALPOST_P, 2) + curState.segment(_ID_GOALPOST_P+2, 2))/2.0;

	Eigen::VectorXd curVel = curState.segment(_ID_V,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;

	double curFacingAngle = getFacingAngleFromLocalState(localState);

	double targetFacingAngle = directionToTheta(curState.segment(_ID_V,2));
	double direction = targetFacingAngle - (curFacingAngle);
	if(direction > M_PI)
		direction = direction - 2*M_PI;
	else if(direction < -M_PI)
		direction = direction + 2*M_PI;

	action[2] = torqueScale*direction;

	action.segment(0,2) = rotate2DVector(action.segment(0,2),  -(curFacingAngle));
	action.segment(0,2).normalize();

	action[3] = hardcodedShootingPower;
	
	return action;
}

bool isBallInPenalty(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd ballPosition = curState.segment(_ID_P,2) + curState.segment(_ID_BALL_P,2);

	return ballPosition[0] < -3.0;
}
bool isNotBallInPenalty(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd ballPosition = curState.segment(_ID_P,2) + curState.segment(_ID_BALL_P,2);

	return ballPosition[0] >= -3.0;
}

Eigen::VectorXd actionBallInPenalty(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd targetPosition = curState.segment(_ID_P,2);
	// targetPosition[0] = 4.0 * (rand()/(double)RAND_MAX );
	// targetPosition[1] = 3.0 * (rand()/(double)RAND_MAX ) - 6.0/2.0;
	// cout<<"@"<<endl;

	double distanceAlly1 = curState.segment(_ID_ALLY1_P, 2).norm();
	double distanceAlly2 = curState.segment(_ID_ALLY2_P, 2).norm();

	Eigen::VectorXd oppAttackerPosition;
	if(distanceAlly1 <= distanceAlly2)
	{
		oppAttackerPosition = curState.segment(_ID_P,2) + curState.segment(_ID_OP_ATK1_P, 2);
	}
	else
	{
		oppAttackerPosition = curState.segment(_ID_P,2) + curState.segment(_ID_OP_ATK2_P, 2);
	}

	double oppAttackerDirection = oppAttackerPosition[1];
	oppAttackerDirection /= abs(oppAttackerDirection);
	targetPosition[1] = -oppAttackerDirection*3.0;

	Eigen::VectorXd targetVel = targetPosition - curState.segment(_ID_P,2);

	Eigen::VectorXd curVel = curState.segment(_ID_V, 2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;

	double curFacingAngle = getFacingAngleFromLocalState(localState);

	double targetFacingAngle = directionToTheta(curState.segment(_ID_V,2));
	double direction = targetFacingAngle - (curFacingAngle);
	if(direction > M_PI)
		direction = direction - 2*M_PI;
	else if(direction < -M_PI)
		direction = direction + 2*M_PI;

	action[2] = torqueScale*direction;

	action.segment(0,2) = rotate2DVector(action.segment(0,2), -curFacingAngle);
	action.segment(0,2).normalize();
	action[3] = -hardcodedShootingPower;
	
	return action;
}

double closeFactor = 0.4;

bool isNotPlayerOnBallToGoal(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = p + (curState.segment(_ID_GOALPOST_P+4, 2) + curState.segment(_ID_GOALPOST_P+6, 2))/2.0;

	Eigen::VectorXd goalpostToBall = (ballP - centerOfGoalpost);
	if(goalpostToBall.norm() > closeFactor)
		goalpostToBall = goalpostToBall.normalized() * closeFactor;
	Eigen::VectorXd targetPosition = centerOfGoalpost + goalpostToBall;

	return (targetPosition - p).norm() >= 0.3 || (ballP - centerOfGoalpost).norm() > 1.8;
}
bool isPlayerOnBallToGoal(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = p + (curState.segment(_ID_GOALPOST_P+4, 2) + curState.segment(_ID_GOALPOST_P+6, 2))/2.0;

	Eigen::VectorXd goalpostToBall = (ballP - centerOfGoalpost);
	double d_goalpostToP = (p - centerOfGoalpost).norm();
	Eigen::VectorXd goalpostToTarget;// = goalpostToBall;
	// if(goalpostToBall.norm() > d_goalpostToP)
	goalpostToTarget = goalpostToBall.normalized() * d_goalpostToP;
	Eigen::VectorXd targetPosition = centerOfGoalpost + goalpostToTarget;

	return (targetPosition - p).norm() < 0.3 && (ballP - centerOfGoalpost).norm() < 1.8;
}

Eigen::VectorXd actionPlayerOnBallToGoal(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = p + (curState.segment(_ID_GOALPOST_P+4, 2) + curState.segment(_ID_GOALPOST_P+6, 2))/2.0;

	Eigen::VectorXd goalpostToBall = (ballP - centerOfGoalpost);
	if(goalpostToBall.norm() > closeFactor)
		goalpostToBall = goalpostToBall.normalized() * closeFactor;
	Eigen::VectorXd targetPosition = centerOfGoalpost + goalpostToBall;

	Eigen::VectorXd targetVel = targetPosition - curState.segment(_ID_P,2);

	Eigen::VectorXd curVel = curState.segment(_ID_V, 2);

	// if(targetVel.norm() !=0)
	targetVel = targetVel*4.0;
	
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;
	action.segment(0,2).normalize();

	double curFacingAngle = getFacingAngleFromLocalState(localState);

	double targetFacingAngle = 0;
	double direction = targetFacingAngle - (curFacingAngle);
	// cout<<direction<<" "<<curFacingAngle<<endl;
	if(direction > M_PI)
		direction = direction - 2*M_PI;
	else if(direction < -M_PI)
		direction = direction + 2*M_PI;

	action[2] = torqueScale*direction;

	action.segment(0,2) = rotate2DVector(action.segment(0,2), -curFacingAngle);
	action[3] = -hardcodedShootingPower;
	// action[3] = 1.0;

	
	return action;
}

bool isNotVelHeadingPlayer(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd p = curState.segment(_ID_P, 2);
	Eigen::VectorXd v = curState.segment(_ID_V, 2);

	double distanceAlly1 = curState.segment(_ID_ALLY1_P, 2).norm();
	double distanceAlly2 = curState.segment(_ID_ALLY2_P, 2).norm();

	Eigen::VectorXd targetVel;
	if(distanceAlly1 <= distanceAlly2)
	{
		targetVel = curState.segment(_ID_ALLY1_P, 2);
	}
	else
	{
		targetVel = curState.segment(_ID_ALLY2_P, 2);
	}

	// Eigen::VectorXd targetVel = curState.segment(_ID_ALLY_P, 2);

	double cosTheta =  v.normalized().dot(targetVel.normalized());
	
	return cosTheta <= 1.0 && curState.segment(_ID_BALL_P,2).norm() < 0.30;
}
Eigen::VectorXd
actionVelHeadingPlayer(Eigen::VectorXd localState)
{
	Eigen::VectorXd curState = localStateToOriginState(localState);
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	double distanceAlly1 = curState.segment(_ID_ALLY1_P, 2).norm();
	double distanceAlly2 = curState.segment(_ID_ALLY2_P, 2).norm();

	Eigen::VectorXd targetVel;
	if(distanceAlly1 <= distanceAlly2)
	{
		targetVel = curState.segment(_ID_ALLY1_P, 2);
	}
	else
	{
		targetVel = curState.segment(_ID_ALLY2_P, 2);
	}

	Eigen::VectorXd curVel = curState.segment(_ID_V,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel- curVel;


	Eigen::VectorXd v = curState.segment(_ID_V, 2);

	Eigen::VectorXd allyP;
	if(distanceAlly1 <= distanceAlly2)
	{
		allyP = curState.segment(_ID_ALLY1_P, 2);

	}
	else
	{
		allyP = curState.segment(_ID_ALLY2_P, 2);
	}


	double cosTheta =  v.normalized().dot(allyP.normalized());

	double curFacingAngle = getFacingAngleFromLocalState(localState);

	double targetFacingAngle = directionToTheta(curState.segment(_ID_V,2));
	double direction = targetFacingAngle - (curFacingAngle);
	if(direction > M_PI)
		direction = direction - 2*M_PI;
	else if(direction < -M_PI)
		direction = direction + 2*M_PI;

	action[2] = torqueScale*direction;

	action.segment(0,2) = rotate2DVector(action.segment(0,2), -curFacingAngle);
	action.segment(0,2).normalize();

	action[3] = -hardcodedShootingPower;
	if(cosTheta>0.95)
		action[3] = hardcodedShootingPower;
	
	return action;
}

// bool isNotBallHeadingPlayer(Eigen::VectorXd curState)
// {
// 	Eigen::VectorXd p = curState.segment(_ID_P,2);
// 	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
// 	Eigen::VectorXd allyP = p + curState.segment(_ID_ALLY_P, 2);


// 	Eigen::VectorXd v =curState.segment(_ID_V, 2);
// 	Eigen::VectorXd ballV = v + curState.segment(_ID_BALL_V, 2);

// 	Eigen::VectorXd ballToAlly = allyP - ballP;

// 	double cosTheta =  ballV.normalized().dot(ballToAlly.normalized());
	
// 	return cosTheta < 0.95 && curState.segment(_ID_BALL_P,2).norm() < 0.23;
// }
// Eigen::VectorXd
// actionBallHeadingPlayer(Eigen::VectorXd curState)
// {
// 	Eigen::VectorXd p = curState.segment(_ID_P,2);
// 	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
// 	Eigen::VectorXd targetVel = (curState.segment(_ID_ALLY_P, 2) + curState.segment(_ID_ALLY_P+2, 2))/2.0;

// 	Eigen::VectorXd curVel = curState.segment(_ID_V,2);

// 	if(targetVel.norm() !=0)
// 		targetVel = targetVel.normalized()*4.0;
// 	Eigen::VectorXd action(4);
// 	action.setZero();
// 	action.segment(0,2) = targetVel - curVel;

// 	Eigen::VectorXd v =curState.segment(_ID_V, 2);
// 	Eigen::VectorXd ballV = v + curState.segment(_ID_BALL_V, 2);
// 	Eigen::VectorXd allyP = p + (curState.segment(_ID_ALLY_P, 2) + curState.segment(_ID_ALLY_P+2, 2))/2.0;

// 	Eigen::VectorXd ballToAlly = allyP - ballP;

// 	double cosTheta =  ballV.normalized().dot(ballToAlly.normalized());
// 	if(cosTheta<0.95)
// 		action[3] = 1.0;
// 	return action;
// }




bool trueReturnFunction(Eigen::VectorXd curState)
{
	return true;
}

BNode* basicPlayer()
{
	BNode* shootRootNode = new BNode("Shooting_Root", BNType::ROOT);

	BNode* infWNode = new BNode("Infinite_While", BNType::WHILE, shootRootNode);
	infWNode->setConditionFunction(trueReturnFunction);

	BNode* shootSNode = new BNode("Shooting_Sequence", BNType::SEQUENCE, infWNode);

		BNode* followWNode = new BNode("Follow_While", BNType::WHILE, shootSNode);
		followWNode->setConditionFunction(isNotCloseToBall);

			BNode* followENode = new BNode("Follow_Execution", BNType::EXECUTION, followWNode);
			followENode->setActionFunction(actionCloseToBall);

		BNode* velWNode = new BNode("Vel_While", BNType::WHILE, shootSNode);
		velWNode->setConditionFunction(isNotVelHeadingGoal);

			BNode* velENode = new BNode("Vel_Exceution", BNType::EXECUTION, velWNode);
			velENode->setActionFunction(actionVelHeadingGoal);

		BNode* shootWNode = new BNode("Shoot_While", BNType::WHILE, shootSNode);
		shootWNode->setConditionFunction(isNotBallHeadingGoal);

			BNode* shootENode = new BNode("Shoot_Exceution", BNType::EXECUTION, shootWNode);
			shootENode->setActionFunction(actionBallHeadingGoal);

	return shootRootNode;
}

BNode* attackerPlayer()
{
	BNode* shootRootNode = new BNode("Shooting_Root", BNType::ROOT);

	BNode* infWNode = new BNode("Infinite_While", BNType::WHILE, shootRootNode);
	infWNode->setConditionFunction(trueReturnFunction);

	BNode* attackerSNode = new BNode("Shooting_Sequence", BNType::SEQUENCE, infWNode);

	// BNode* attackerINode = new BNode("Position_If", BNType::IF, shootSNode);
	// 	attackerINode->setConditionFunction(isBallOverPenalty);

		BNode* attackerWNode = new BNode("Position_While", BNType::WHILE, attackerSNode);
		attackerWNode->setConditionFunction(isBallInPenalty);

			BNode* attackerENode = new BNode("Position_Excecution", BNType::EXECUTION, attackerWNode);
			attackerENode->setActionFunction(actionBallInPenalty);

		BNode* attackerIFNode = new BNode("Position_While_false", BNType::IF, attackerSNode);
		attackerIFNode->setConditionFunction(isNotBallInPenalty);

			BNode* shootSNode = new BNode("Shooting_Sequence", BNType::SEQUENCE, attackerIFNode);

				BNode* followWNode = new BNode("Follow_While", BNType::WHILE, shootSNode);
				followWNode->setConditionFunction(isNotCloseToBall);

					BNode* followENode = new BNode("Follow_Execution", BNType::EXECUTION, followWNode);
					followENode->setActionFunction(actionCloseToBall);

				BNode* velWNode = new BNode("Vel_While", BNType::WHILE, shootSNode);
				velWNode->setConditionFunction(isNotVelHeadingGoal);

					BNode* velENode = new BNode("Vel_Exceution", BNType::EXECUTION, velWNode);
					velENode->setActionFunction(actionVelHeadingGoal);

				BNode* shootWNode = new BNode("Shoot_While", BNType::WHILE, shootSNode);
				shootWNode->setConditionFunction(isNotBallHeadingGoal);

					BNode* shootENode = new BNode("Shoot_Exceution", BNType::EXECUTION, shootWNode);
					shootENode->setActionFunction(actionBallHeadingGoal);

	return shootRootNode;
}

BNode* defenderPlayer()
{
	BNode* shootRootNode = new BNode("Defending_Root", BNType::ROOT);

	BNode* infWNode = new BNode("Infinite_While", BNType::WHILE, shootRootNode);
	infWNode->setConditionFunction(trueReturnFunction);

	BNode* defenderSNode = new BNode("Defending_Sequence", BNType::SEQUENCE, infWNode);

		BNode* defenderWNode = new BNode("Position_While", BNType::WHILE, defenderSNode);
		defenderWNode->setConditionFunction(isNotPlayerOnBallToGoal);

			BNode* defenderENode = new BNode("Position_Excecution", BNType::EXECUTION, defenderWNode);
			defenderENode->setActionFunction(actionPlayerOnBallToGoal);

		BNode* defenderIFNode = new BNode("Position_IF_FALSE", BNType::IF, defenderSNode);
		defenderIFNode->setConditionFunction(isPlayerOnBallToGoal);

			BNode* passSNode = new BNode("Pass_Sequence", BNType::SEQUENCE, defenderIFNode);

				BNode* followWNode = new BNode("Follow_While", BNType::WHILE, passSNode);
				followWNode->setConditionFunction(isNotCloseToBall_soft);

					BNode* followENode = new BNode("Follow_Execution", BNType::EXECUTION, followWNode);
					followENode->setActionFunction(actionCloseToBall);

				BNode* passVelWNode = new BNode("PassVel_While", BNType::WHILE, passSNode);
				passVelWNode->setConditionFunction(isNotVelHeadingPlayer);

					BNode* velENode = new BNode("PassVel_Exceution", BNType::EXECUTION, passVelWNode);
					velENode->setActionFunction(actionVelHeadingPlayer);

				// BNode* passWNode = new BNode("Pass_While", BNType::WHILE, passSNode);
				// passWNode->setConditionFunction(isNotBallHeadingPlayer);

				// 	BNode* passENode = new BNode("Pass_Exceution", BNType::EXECUTION, passWNode);
				// 	passENode->setActionFunction(actionBallHeadingPlayer);

	return shootRootNode;
}

void
Environment::
initBehaviorTree()
{
	//Basic ball chasing
	if(mNumChars == 6)
	{
		mBTs[0] = defenderPlayer();
		mBTs[1] = attackerPlayer();
		mBTs[2] = attackerPlayer();

		// mBTs[0] = basicPlayer();
		// mBTs[1] = basicPlayer();


		mBTs[3] = defenderPlayer();
		mBTs[4] = attackerPlayer();
		mBTs[5] = attackerPlayer();

		// mBTs[2] = basicPlayer();
		// mBTs[3] = basicPlayer();
	}



	if(mNumChars == 4)
	{
		mBTs[0] = defenderPlayer();
		mBTs[1] = attackerPlayer();

		// mBTs[0] = basicPlayer();
		// mBTs[1] = basicPlayer();


		mBTs[2] = defenderPlayer();
		mBTs[3] = attackerPlayer();

		// mBTs[2] = basicPlayer();
		// mBTs[3] = basicPlayer();
	}
	if(mNumChars == 2)
	{
		mBTs[0] = basicPlayer();
		mBTs[1] = basicPlayer();
		// mBTs[1] = attackerPlayer();

	}
}

Eigen::VectorXd
Environment::
getActionFromBTree(int index)
{
	// Eigen::VectorXd state = getState(index);
	// cout<<mStates[index].size()<<endl;
	// cout<<"Facing vel : "<<mFacingVels[0]<<" "<<mFacingVels[1]<<" "<<mFacingVels[2]<<" "<<mFacingVels[3]<<endl;

	return mBTs[index]->getActionFromBTree(mLocalStates[index]);
}

// void
// Environment::
// setHardcodedAction(int index)
// {
// 	for(int i=0;i<4;i++)
// 	{
// 		setAction(i, getActionFromBTree(i));
// 	}
// }


// std::vector<int> 
// Environment::
// getAgentViewImg(int index)
// {

// }

void
Environment::
reconEnvFromState(int index, Eigen::VectorXd curLocalState)
{
	Eigen::VectorXd curState = localStateToOriginState(curLocalState, mNumChars);

	double facingAngle = getFacingAngleFromLocalState(curLocalState);


	double reviseStateByTeam = -1;
	if(mCharacters[index]->getTeamName() == mGoalposts[0].first)
	{
		reviseStateByTeam = 1;
	}
	facingAngle = M_PI * (1-reviseStateByTeam)/2.0 + facingAngle;
	// this->getCharacter(index)->getSkeleton()->setPosition(2, facingAngle);



	Eigen::Vector2d p = curState.segment(_ID_P, 2);
	Eigen::Vector2d v = curState.segment(_ID_V, 2);
	Eigen::Vector2d temp;

	if(mNumChars == 6)
	{

		if(index%3 == 0)
		{
			this->getCharacter(0)->getSkeleton()->setPosition(0, 8.0*curState[_ID_P]);
			this->getCharacter(0)->getSkeleton()->setPosition(1, 8.0*curState[_ID_P+1]);

			this->getCharacter(0)->getSkeleton()->setVelocity(0, 8.0*curState[_ID_V]);
			this->getCharacter(0)->getSkeleton()->setVelocity(1, 8.0*curState[_ID_V+1]);

			temp = curState.segment(_ID_ALLY1_P, 2);
			this->getCharacter(1)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
			this->getCharacter(1)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

			temp = curState.segment(_ID_ALLY1_V, 2);
			this->getCharacter(1)->getSkeleton()->setVelocity(0, 8.0*(v+temp)[0]);
			this->getCharacter(1)->getSkeleton()->setVelocity(1, 8.0*(v+temp)[1]);
		}
		else
		{

			this->getCharacter(1)->getSkeleton()->setPosition(0, 8.0*curState[_ID_P]);
			this->getCharacter(1)->getSkeleton()->setPosition(1, 8.0*curState[_ID_P+1]);

			this->getCharacter(1)->getSkeleton()->setVelocity(0, 8.0*curState[_ID_V]);
			this->getCharacter(1)->getSkeleton()->setVelocity(1, 8.0*curState[_ID_V+1]);

			temp = curState.segment(_ID_ALLY1_P, 2);
			this->getCharacter(0)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
			this->getCharacter(0)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

			temp = curState.segment(_ID_ALLY1_V, 2);
			this->getCharacter(0)->getSkeleton()->setVelocity(0, 8.0*(v+temp)[0]);
			this->getCharacter(0)->getSkeleton()->setVelocity(1, 8.0*(v+temp)[1]);
		}

		temp = curState.segment(_ID_ALLY2_P, 2);
		this->getCharacter(2)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
		this->getCharacter(2)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

		temp = curState.segment(_ID_ALLY2_V, 2);
		this->getCharacter(2)->getSkeleton()->setVelocity(0, 8.0*(v+temp)[0]);
		this->getCharacter(2)->getSkeleton()->setVelocity(1, 8.0*(v+temp)[1]);

		temp = curState.segment(_ID_OP_DEF_P, 2);
		this->getCharacter(3)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
		this->getCharacter(3)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

		temp = curState.segment(_ID_OP_DEF_V, 2);
		this->getCharacter(3)->getSkeleton()->setVelocity(0, 8.0*(v+temp)[0]);
		this->getCharacter(3)->getSkeleton()->setVelocity(1, 8.0*(v+temp)[1]);

		temp = curState.segment(_ID_OP_ATK1_P, 2);
		this->getCharacter(4)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
		this->getCharacter(4)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

		temp = curState.segment(_ID_OP_ATK1_V, 2);
		this->getCharacter(4)->getSkeleton()->setVelocity(0, 8.0*(v+temp)[0]);
		this->getCharacter(4)->getSkeleton()->setVelocity(1, 8.0*(v+temp)[1]);


		temp = curState.segment(_ID_OP_ATK2_P, 2);
		this->getCharacter(5)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
		this->getCharacter(5)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

		temp = curState.segment(_ID_OP_ATK2_V, 2);
		this->getCharacter(5)->getSkeleton()->setVelocity(0, 8.0*(v+temp)[0]);
		this->getCharacter(5)->getSkeleton()->setVelocity(1, 8.0*(v+temp)[1]);
	}
	else if(mNumChars == 2)
	{

		this->getCharacter(0)->getSkeleton()->setPosition(0, 8.0*curState[_ID_P]);
		this->getCharacter(0)->getSkeleton()->setPosition(1, 8.0*curState[_ID_P+1]);

		this->getCharacter(0)->getSkeleton()->setVelocity(0, 8.0*curState[_ID_V]);
		this->getCharacter(0)->getSkeleton()->setVelocity(1, 8.0*curState[_ID_V+1]);

		temp = curState.segment(_ID_OP_P, 2);
		this->getCharacter(1)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
		this->getCharacter(1)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

		temp = curState.segment(_ID_OP_V, 2);
		this->getCharacter(1)->getSkeleton()->setVelocity(0, 8.0*(v+temp)[0]);
		this->getCharacter(1)->getSkeleton()->setVelocity(1, 8.0*(v+temp)[1]);

	}
	if(mNumChars == 4)
	{
		if(index == 0 || index == 1)
		{
			this->getCharacter(0)->getSkeleton()->setPosition(0, 8.0*curState[_ID_P]);
			this->getCharacter(0)->getSkeleton()->setPosition(1, 8.0*curState[_ID_P+1]);

			this->getCharacter(0)->getSkeleton()->setVelocity(0, 8.0*curState[_ID_V]);
			this->getCharacter(0)->getSkeleton()->setVelocity(1, 8.0*curState[_ID_V+1]);

			temp = curState.segment(_ID_ALLY_P, 2);
			this->getCharacter(1)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
			this->getCharacter(1)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

			temp = curState.segment(_ID_ALLY_V, 2);
			this->getCharacter(1)->getSkeleton()->setVelocity(0, 8.0*(v+temp)[0]);
			this->getCharacter(1)->getSkeleton()->setVelocity(1, 8.0*(v+temp)[1]);

			temp = curState.segment(_ID_OP_DEF_P, 2);
			this->getCharacter(2)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
			this->getCharacter(2)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

			temp = curState.segment(_ID_OP_DEF_V, 2);
			this->getCharacter(2)->getSkeleton()->setVelocity(0, 8.0*(v+temp)[0]);
			this->getCharacter(2)->getSkeleton()->setVelocity(1, 8.0*(v+temp)[1]);

			temp = curState.segment(_ID_OP_ATK_P, 2);
			this->getCharacter(3)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
			this->getCharacter(3)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

			temp = curState.segment(_ID_OP_ATK_V, 2);
			this->getCharacter(3)->getSkeleton()->setVelocity(0, 8.0*(v+temp)[0]);
			this->getCharacter(3)->getSkeleton()->setVelocity(1, 8.0*(v+temp)[1]);


			temp = curState.segment(_ID_BALL_P, 2);
			this->ballSkel->setPosition(0, 8.0*(p+temp)[0]);
			this->ballSkel->setPosition(1, 8.0*(p+temp)[1]);

			temp = curState.segment(_ID_BALL_V, 2);
			this->ballSkel->setVelocity(0, 8.0*(v+temp)[0]);
			this->ballSkel->setVelocity(1, 8.0*(v+temp)[1]);
		}
		else
		{
			this->getCharacter(2)->getSkeleton()->setPosition(0, -8.0*curState[_ID_P]);
			this->getCharacter(2)->getSkeleton()->setPosition(1, -8.0*curState[_ID_P+1]);

			this->getCharacter(2)->getSkeleton()->setVelocity(0, -8.0*curState[_ID_V]);
			this->getCharacter(2)->getSkeleton()->setVelocity(1, -8.0*curState[_ID_V+1]);

			temp = curState.segment(_ID_ALLY_P, 2);
			this->getCharacter(3)->getSkeleton()->setPosition(0, -8.0*(p+temp)[0]);
			this->getCharacter(3)->getSkeleton()->setPosition(1, -8.0*(p+temp)[1]);

			temp = curState.segment(_ID_ALLY_V, 2);
			this->getCharacter(3)->getSkeleton()->setVelocity(0, -8.0*(v+temp)[0]);
			this->getCharacter(3)->getSkeleton()->setVelocity(1, -8.0*(v+temp)[1]);

			temp = curState.segment(_ID_OP_DEF_P, 2);
			this->getCharacter(0)->getSkeleton()->setPosition(0, -8.0*(p+temp)[0]);
			this->getCharacter(0)->getSkeleton()->setPosition(1, -8.0*(p+temp)[1]);

			temp = curState.segment(_ID_OP_DEF_V, 2);
			this->getCharacter(0)->getSkeleton()->setVelocity(0, -8.0*(v+temp)[0]);
			this->getCharacter(0)->getSkeleton()->setVelocity(1, -8.0*(v+temp)[1]);

			temp = curState.segment(_ID_OP_ATK_P, 2);
			this->getCharacter(1)->getSkeleton()->setPosition(0, -8.0*(p+temp)[0]);
			this->getCharacter(1)->getSkeleton()->setPosition(1, -8.0*(p+temp)[1]);

			temp = curState.segment(_ID_OP_ATK_V, 2);
			this->getCharacter(1)->getSkeleton()->setVelocity(0, -8.0*(v+temp)[0]);
			this->getCharacter(1)->getSkeleton()->setVelocity(1, -8.0*(v+temp)[1]);

			temp = curState.segment(_ID_BALL_P, 2);
			this->ballSkel->setPosition(0, -8.0*(p+temp)[0]);
			this->ballSkel->setPosition(1, -8.0*(p+temp)[1]);

			temp = curState.segment(_ID_BALL_V, 2);
			this->ballSkel->setVelocity(0, -8.0*(v+temp)[0]);
			this->ballSkel->setVelocity(1, -8.0*(v+temp)[1]);

		}
		
	}

	// temp = curState.segment(_ID_BALL_P, 2);
	// this->ballSkel->setPosition(0, 8.0*(p+temp)[0]);
	// this->ballSkel->setPosition(1, 8.0*(p+temp)[1]);

	// temp = curState.segment(_ID_BALL_V, 2);
	// this->ballSkel->setVelocity(0, 8.0*(v+temp)[0]);
	// this->ballSkel->setVelocity(1, 8.0*(v+temp)[1]);
}