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
mIsTerminalState(false), mTimeElapsed(0), mNumIterations(0)
{
	srand((unsigned int)time(0));
	initCharacters();
	initGoalposts();
	initFloor();
	initBall();
	getNumState();
	mWorld->getConstraintSolver()->removeAllConstraints();
	mWindow = new AgentEnvWindow(0, this);
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
		mAction.resize(zeroVel.rows()+touch.rows());
		mAction << zeroVel, touch;
		mActions.push_back(mAction);
	}


	mStates.resize(mNumChars);
	mForces.resize(mNumChars);
	mBTs.resize(mNumChars);
	initBehaviorTree();
}


void 
Environment::
resetCharacterPositions()
{
	if(mNumChars == 4)
	{
		std::vector<Eigen::Vector3d> charPositions;
		charPositions.push_back(Eigen::Vector3d(-1.0, 0.5, 0.0));
		charPositions.push_back(Eigen::Vector3d(-1.0, -0.5, 0.0));
		charPositions.push_back(Eigen::Vector3d(1.0, 0.5, M_PI));
		charPositions.push_back(Eigen::Vector3d(1.0, -0.5, M_PI));

		for(int i=0;i<mNumChars;i++)
		{
			mCharacters[i]->getSkeleton()->setPositions(charPositions[i]);
			mCharacters[i]->getSkeleton()->setVelocities(Eigen::Vector3d(0.0, 0.0, 0.0));
		}
	}
	else if(mNumChars == 2)
	{
		std::vector<Eigen::Vector3d> charPositions;
		charPositions.push_back(Eigen::Vector3d(-1.0, 0.0, 0.0));
		charPositions.push_back(Eigen::Vector3d(1.0, 0.0, M_PI));

		for(int i=0;i<mNumChars;i++)
		{
			mCharacters[i]->getSkeleton()->setPositions(charPositions[i]);
			mCharacters[i]->getSkeleton()->setVelocities(Eigen::Vector3d(0.0, 0.0, 0.0));
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

	for(int i=0;i<collidingWalls.size();i++)
	{
		switch(collidingWalls[i])
		{
			case 0:
			if(me*skel->getVelocity(0)>0)
				skel->setVelocity(0, -me*skel->getVelocity(0));
				skel->setForce(0, 0);
			break;
			case 1:
			if(me*skel->getVelocity(0)<0)
				skel->setVelocity(0, -me*skel->getVelocity(0));
				skel->setForce(0, 0);
			break;
			case 2:
			if(me*skel->getVelocity(1)>0)
				skel->setVelocity(1, -me*skel->getVelocity(1));
				skel->setForce(1, 0);
			break;
			case 3:
			if(me*skel->getVelocity(1)<0)
				skel->setVelocity(1, -me*skel->getVelocity(1));
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

	double kickPower = mActions[index][3];

	// if(kickPower>=1)
	// 	kickPower = 1;

	if(kickPower > 0)
	{
		kickPower = 1.0;
		// kickPower = 1.0/(exp(-kickPower)+1);
		// cout<<"Kicked!"<<endl;
		// kickPower = 1.0;
		ballSkel->setVelocities(skel->getVelocities().segment(0,2)*(2.0+me)*(1.0*kickPower));
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

		Eigen::Vector3d skel1VelocitiesFull = skel1->getVelocities();
		Eigen::Vector3d skel2VelocitiesFull = skel1->getVelocities();
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
		for(int j=0;j<i;j++)
		{
			handlePlayerContact(i,j,me);
		}
	}
}

void 
Environment::
boundBallVelocitiy(double maxVel)
{
	Eigen::VectorXd ballVel = ballSkel->getVelocities();
	// cout<<"ballVel size: "<<ballVel.size()<<endl;
	for(int i=0;i<ballVel.size();i++)
	{
		// cout<<"i:"<<i<<endl;
		if(abs(ballVel[i])>maxVel)
		{
			ballVel[i] = ballVel[i]/abs(ballVel[i])*maxVel;
		}
	}
	ballSkel->setVelocities(ballVel);	
}

void 
Environment::
dampBallVelocitiy(double dampPower)
{
	Eigen::VectorXd ballForce = ballSkel->getForces();
	ballForce -= dampPower*ballSkel->getVelocities();
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

	// handlePlayerContacts(0.7);

	for(int i=0;i<mCharacters.size();i++)
	{
		handleWallContact(mCharacters[i]->getSkeleton(), 0.08, 0.5);


		if(mStates[i][_ID_KICKABLE] == 1)
		{
			// cout<<"here right?"<<endl;
			handleBallContact(i, 0.12, 2.0);
		}
	}

	handleWallContact(ballSkel, 0.08, 0.8);

	boundBallVelocitiy(8.0);
	dampBallVelocitiy(2.0);

	mWorld->step();
}


void
Environment::
stepAtOnce()
{
	int sim_per_control = this->getSimulationHz()/this->getControlHz();
	for(int i=0;i<sim_per_control;i++)
	{
		this->step();
	}
	mWindow->display();
}


Eigen::VectorXd
Environment::
getState(int index)
{
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
	if(relativeBallP.norm()<0.15+0.08)
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
			SkeletonPtr skel = mCharacters[i]->getSkeleton();
			otherS.segment(count*4,2) = skel->getPositions().segment(0,2) - p;
			otherS.segment(count*4+2,2) = skel->getVelocities().segment(0,2) - v;
			count++;
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

	Eigen::VectorXd state;

	state.resize(p.rows() + v.rows() + ballP.rows() + ballV.rows() + kickable.rows() + simpleGoalpostPositions.rows()
		+ otherS.rows());

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

	mStates[index] = state;
	return normalizeNNState(state);
}



double
Environment::
getReward(int index)
{
	double reward = 0.0;
	Eigen::VectorXd p = mStates[index].segment(_ID_P,2);
	Eigen::VectorXd ballP = p + mStates[index].segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = p + (mStates[index].segment(_ID_GOALPOST_P, 2) + mStates[index].segment(_ID_GOALPOST_P+2, 2))/2.0;


	Eigen::VectorXd v = mStates[index].segment(_ID_V, 2);
	Eigen::VectorXd ballV = v + mStates[index].segment(_ID_BALL_V, 2);

	Eigen::VectorXd ballToGoalpost = centerOfGoalpost - ballP;

	// reward = 0.1 * exp(-(p-ballP).norm());

	// reward -= 3.0 * exp(-1/ballToGoalpost.norm());

	if(ballV.norm()>0 && ballToGoalpost.norm()>0)
	{
		// double cosTheta =  ballV.normalized().dot(ballToGoalpost.normalized());
		// reward += ballV.norm()* exp(-2.0 * acos(cosTheta));

		// Eigen::VectorXd nextP = ballP + ballV * 1.0 / 30;
		// Eigen::VectorXd diff = (nextP - ballP) - (p - ballP);
		// reward = 0.1 * exp(-(p-ballP).norm());
		// reward += ballV.dot(ballToGoalpost.normalized());
	}
	// else
	// 	reward += 0;



	// goal Reward
	double ballRadius = 0.1;
	double goalpostSize = 1.5;
	Eigen::Vector2d ballPosition = ballSkel->getPositions();
	Eigen::Vector2d widthVector = Eigen::Vector2d::UnitX();
	Eigen::Vector2d heightVector = Eigen::Vector2d::UnitY();



	if(widthVector.dot(ballPosition)+ballRadius >= mGoalposts[1].second.x())
	{
		if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
		{
			// std::cout<<"Red Team GOALL!!"<<std::endl;
			// if(!goalRewardPaid[index])

			if(mCharacters[index]->getTeamName() == "A")
				reward += 200;
			else
				reward -= 200;
			// goalRewardPaid[index] = true;
			mIsTerminalState = true;
		}
	}
	else if(widthVector.dot(ballPosition)-ballRadius < mGoalposts[0].second.x())
	{
		if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
		{
			// std::cout<<"Blue Team GOALL!!"<<std::endl;
			// if(!goalRewardPaid[index])
			if(mCharacters[index]->getTeamName() == "A")
				reward -= 200;
			else
				reward += 200;
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
		// getLinearActorReward(i);
		if (i==0)
		{
			// rewards.push_back();
			mAccScore[i] += getReward(i);
		}

	}

	return rewards;
}

void
Environment::
applyAction(int index)
{
	// double maxVel = 4.0;
	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	skel->setForces(mForces[index]);

	double reviseStateByTeam = -1;


	if( mCharacters[index]->getTeamName() == mGoalposts[0].first)
		reviseStateByTeam = 1;

	Eigen::VectorXd vel = reviseStateByTeam * skel->getVelocities();

	if (vel.norm() > maxVel)
		vel = vel/vel.norm() * maxVel;
	skel->setVelocities(reviseStateByTeam * vel);
}

void 
Environment::
setAction(int index, const Eigen::VectorXd& a)
{
	bool isNanOccured = false;
	// double maxVel = 4.0;

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


	// for(int i=0;i<2;i++)
	// {
	// 	if(mActions[index][i] > 1.0)
	// 		mActions[index][i] = 1.0;
	// 	if(mActions[index][i] < -1.0)
	// 		mActions[index][i] = -1.0;
	// }

	if(mActions[index].segment(0,3).norm()>1.0)
	{
		mActions[index].segment(0,3) /= mActions[index].segment(0,2).norm();
	}

	// if(index == 0)
	// 	mActions[index][2] = 1.0;

	// if(index != 0)
	// {
	// 	mActions[index][3] = -1;
	// }

	Eigen::VectorXd applyingForce = mActions[index].segment(0,3);

	// applyingForce /= applyingForce.norm();

	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	
	double reviseStateByTeam = -1;
	if( mCharacters[index]->getTeamName() == mGoalposts[0].first)
		reviseStateByTeam = 1;

	// Eigen::VectorXd vel = reviseStateByTeam * skel->getVelocities();

	mForces[index] = 200.0*reviseStateByTeam*applyingForce;
}


bool
Environment::
isTerminalState()
{
	if(mTimeElapsed>10000.0)
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
	ballPosition[0] = 4.0 * (rand()/(double)RAND_MAX ) - 4.0/2.0;
	ballPosition[1] = 3.0 * (rand()/(double)RAND_MAX ) - 3.0/2.0;
	ballSkel->setPositions(ballPosition);
	Eigen::VectorXd ballVel = ballSkel->getVelocities();
	ballVel[0] = 8.0 * (rand()/(double)RAND_MAX ) - 4.0;
	ballVel[1] = 8.0 * (rand()/(double)RAND_MAX ) - 4.0;
	ballSkel->setVelocities(ballVel);

	for(int i=0;i<mNumChars;i++)
	{
		SkeletonPtr skel = mCharacters[i]->getSkeleton();
		Eigen::VectorXd skelPosition = skel->getPositions();
		skelPosition[0] = 6.0 * (rand()/(double)RAND_MAX ) - 6.0/2.0;
		skelPosition[1] = 4.5 * (rand()/(double)RAND_MAX ) - 4.5/2.0;
		skelPosition[2] = 2.0 * M_PI * (rand()/(double)RAND_MAX );
		skel->setPositions(skelPosition);
		Eigen::VectorXd skelVel = skel->getVelocities();
		skelVel[0] = 3.0 * (rand()/(double)RAND_MAX ) - 1.5;
		skelVel[1] = 3.0 * (rand()/(double)RAND_MAX ) - 1.5;
		skelVel[2] = 0.0;
		skel->setVelocities(skelVel);
		// if(i == 0 && rand()%2 == 0)
		// {
		// 	ballSkel->setPositions(skelPosition);
		// 	ballSkel->setVelocities(skelVel);
		// }
	}


	mIsTerminalState = false;
	mTimeElapsed = 0;

	mAccScore.setZero();


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
	else if(centerVector.dot(widthVector) <= -(groundWidth-radius))
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
	normalizedState.segment(9, numState-9) = state.segment(9, numState-9)/8.0;
	return normalizedState;
}


Eigen::VectorXd
Environment::
unNormalizeNNState(Eigen::VectorXd outSubgoal)
{
	Eigen::VectorXd scaledSubgoal = outSubgoal;
	int numState = scaledSubgoal.size();
	scaledSubgoal.segment(0, 8) = outSubgoal.segment(0,8)*8.0;
	scaledSubgoal.segment(9, numState-9) = outSubgoal.segment(9, numState-9)*8.0;
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
	return curState.segment(_ID_BALL_P,2).norm() >= 0.05;
}

Eigen::VectorXd
actionCloseToBall(Eigen::VectorXd curState)
{
	Eigen::VectorXd curVel = curState.segment(_ID_V,2);
	Eigen::VectorXd targetVel = curState.segment(_ID_BALL_P,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;

	return action;
}

Eigen::VectorXd
actionCloseToBallWithShooting(Eigen::VectorXd curState)
{
	Eigen::VectorXd curVel = curState.segment(_ID_V,2);
	Eigen::VectorXd targetVel = curState.segment(_ID_BALL_P,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;

	action[3] = 1.0;

	return action;
}
bool isNotVelHeadingGoal(Eigen::VectorXd curState)
{
	Eigen::VectorXd p = curState.segment(_ID_P, 2);
	Eigen::VectorXd v = curState.segment(_ID_V, 2);
	Eigen::VectorXd agentToGoalpost = (curState.segment(_ID_GOALPOST_P, 2) + curState.segment(_ID_GOALPOST_P+2, 2))/2.0;

	double cosTheta =  v.normalized().dot(agentToGoalpost.normalized());
	
	return cosTheta < 0.95 && curState.segment(_ID_BALL_P,2).norm() < 0.23;
}
Eigen::VectorXd
actionVelHeadingGoal(Eigen::VectorXd curState)
{
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd targetVel = (curState.segment(_ID_GOALPOST_P, 2) + curState.segment(_ID_GOALPOST_P+2, 2))/2.0;

	Eigen::VectorXd curVel = curState.segment(_ID_V,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel- curVel;

	action[3] = 0.0;
	return action;
}

bool isNotBallHeadingGoal(Eigen::VectorXd curState)
{
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
actionBallHeadingGoal(Eigen::VectorXd curState)
{
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd targetVel = (curState.segment(_ID_GOALPOST_P, 2) + curState.segment(_ID_GOALPOST_P+2, 2))/2.0;

	Eigen::VectorXd curVel = curState.segment(_ID_V,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;

	action[3] = 1.0;
	return action;
}

bool isBallInPenalty(Eigen::VectorXd curState)
{
	Eigen::VectorXd ballPosition = curState.segment(_ID_P,2) + curState.segment(_ID_BALL_P,2);

	return ballPosition[0] < -2.0;
}
bool isNotBallInPenalty(Eigen::VectorXd curState)
{
	Eigen::VectorXd ballPosition = curState.segment(_ID_P,2) + curState.segment(_ID_BALL_P,2);

	return ballPosition[0] >= -2.0;
}

Eigen::VectorXd actionNotBallInPenalty(Eigen::VectorXd curState)
{
	Eigen::VectorXd targetPosition = curState.segment(_ID_P,2);
	targetPosition[0] = 4.0 * (rand()/(double)RAND_MAX );
	targetPosition[1] = 3.0 * (rand()/(double)RAND_MAX ) - 6.0/2.0;

	Eigen::VectorXd targetVel = targetPosition - curState.segment(_ID_P,2);

	Eigen::VectorXd curVel = curState.segment(_ID_V, 2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;

	return action;
}

bool isNotPlayerOnBallToGoal(Eigen::VectorXd curState)
{
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = p + (curState.segment(_ID_GOALPOST_P+4, 2) + curState.segment(_ID_GOALPOST_P+6, 2))/2.0;

	Eigen::VectorXd goalpostToBall = (ballP - centerOfGoalpost);
	if(goalpostToBall.norm() > 0.5)
		goalpostToBall = goalpostToBall.normalized() * 0.5;
	Eigen::VectorXd targetPosition = centerOfGoalpost + goalpostToBall;

	return (targetPosition - p).norm() >= 0.5;
}
bool isPlayerOnBallToGoal(Eigen::VectorXd curState)
{
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = p + (curState.segment(_ID_GOALPOST_P+4, 2) + curState.segment(_ID_GOALPOST_P+6, 2))/2.0;

	Eigen::VectorXd goalpostToBall = (ballP - centerOfGoalpost);
	if(goalpostToBall.norm() > 0.5)
		goalpostToBall = goalpostToBall.normalized() * 0.5;
	Eigen::VectorXd targetPosition = centerOfGoalpost + goalpostToBall;

	return (targetPosition - p).norm() < 1.5 && (ballP - targetPosition).norm() < 1.5;
}

Eigen::VectorXd actionPlayerOnBallToGoal(Eigen::VectorXd curState)
{
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = p + (curState.segment(_ID_GOALPOST_P+4, 2) + curState.segment(_ID_GOALPOST_P+6, 2))/2.0;

	Eigen::VectorXd goalpostToBall = (ballP - centerOfGoalpost);
	if(goalpostToBall.norm() > 0.5)
		goalpostToBall = goalpostToBall.normalized() * 0.5;
	Eigen::VectorXd targetPosition = centerOfGoalpost + goalpostToBall;

		Eigen::VectorXd targetVel = targetPosition - curState.segment(_ID_P,2);

	Eigen::VectorXd curVel = curState.segment(_ID_V, 2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;
	// action[3] = 1.0;

	return action;
}

bool isNotVelHeadingPlayer(Eigen::VectorXd curState)
{
	Eigen::VectorXd p = curState.segment(_ID_P, 2);
	Eigen::VectorXd v = curState.segment(_ID_V, 2);
	Eigen::VectorXd agentToGoalpost = (curState.segment(_ID_GOALPOST_P, 2) + curState.segment(_ID_GOALPOST_P+2, 2))/2.0;

	double cosTheta =  v.normalized().dot(agentToGoalpost.normalized());
	
	return cosTheta < 0.9 && curState.segment(_ID_BALL_P,2).norm() < 0.23;
}
Eigen::VectorXd
actionVelHeadingPlayer(Eigen::VectorXd curState)
{
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd targetVel = (curState.segment(_ID_GOALPOST_P, 2) + curState.segment(_ID_GOALPOST_P+2, 2))/2.0;

	Eigen::VectorXd curVel = curState.segment(_ID_V,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel- curVel;

	action[3] = 1.0;
	return action;
}

bool isNotBallHeadingPlayer(Eigen::VectorXd curState)
{
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
actionBallHeadingPlayer(Eigen::VectorXd curState)
{
	Eigen::VectorXd p = curState.segment(_ID_P,2);
	Eigen::VectorXd ballP = p + curState.segment(_ID_BALL_P,2);
	Eigen::VectorXd targetVel = (curState.segment(_ID_GOALPOST_P, 2) + curState.segment(_ID_GOALPOST_P+2, 2))/2.0;

	Eigen::VectorXd curVel = curState.segment(_ID_V,2);

	if(targetVel.norm() !=0)
		targetVel = targetVel.normalized()*4.0;
	Eigen::VectorXd action(4);
	action.setZero();
	action.segment(0,2) = targetVel - curVel;

	action[3] = 1.0;
	return action;
}




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
			attackerENode->setActionFunction(actionNotBallInPenalty);

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

		BNode* defenderIFNode = new BNode("Position_While_false", BNType::IF, defenderSNode);
		defenderIFNode->setConditionFunction(isPlayerOnBallToGoal);

			BNode* passSNode = new BNode("Pass_Sequence", BNType::SEQUENCE, defenderIFNode);

				BNode* followWNode = new BNode("Follow_While", BNType::WHILE, passSNode);
				followWNode->setConditionFunction(isNotCloseToBall);

					BNode* followENode = new BNode("Follow_Execution", BNType::EXECUTION, followWNode);
					followENode->setActionFunction(actionCloseToBallWithShooting);

				// BNode* passVelWNode = new BNode("PassVel_While", BNType::WHILE, passSNode);
				// passVelWNode->setConditionFunction(isNotVelHeadingPlayer);

				// 	BNode* velENode = new BNode("PassVel_Exceution", BNType::EXECUTION, passVelWNode);
				// 	velENode->setActionFunction(actionVelHeadingPlayer);

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

	if(mNumChars == 4)
	{
		mBTs[0] = defenderPlayer();
		mBTs[1] = defenderPlayer();

		// mBTs[2] = defenderPlayer();
		// mBTs[3] = attackerPlayer();

		mBTs[2] = defenderPlayer();
		mBTs[3] = attackerPlayer();
	}
	if(mNumChars == 2)
	{
		mBTs[0] = basicPlayer();
		// mBTs[1] = attackerPlayer();

	}
}

Eigen::VectorXd
Environment::
getActionFromBTree(int index)
{
	Eigen::VectorXd state = getState(index);
	return mBTs[index]->getActionFromBTree(unNormalizeNNState(state));
}

// std::vector<int> 
// Environment::
// getAgentViewImg(int index)
// {

// }
