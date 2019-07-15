#include "Environment.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include <iostream>
#include <chrono>
#include <random>
#include <ctime>

using namespace std;
using namespace dart;
using namespace dart::dynamics;

Environment::
Environment(int control_Hz, int simulation_Hz, int numChars)
:mControlHz(control_Hz), mSimulationHz(simulation_Hz), mNumChars(numChars), mWorld(std::make_shared<dart::simulation::World>()),
mIsTerminalState(false), mTimeElapsed(0)
{
	srand((unsigned int)time(0));
	initCharacters();
	initGoalposts();
	initFloor();
	initBall();
}

	// Create A team, B team players.
void
Environment::
initCharacters()
{
	for(int i=0;i<1;i++)
	{
		mCharacters.push_back(new Character2D("A_" + to_string(i)));
	}
	for(int i=0;i<1;i++)
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
		Eigen::VectorXd zeroVel(mCharacters[i]->getSkeleton()->getNumDofs());
		zeroVel.setZero();
		// Eigen::VectorXd controlBall(2);
		// Eigen::VectorXd mAction(zeroVel.size()+controlBall.size());
		// mAction << zeroVel, controlBall;
		mActions.push_back(zeroVel);
	}

	mKicked.resize(4);
	mKicked.setZero();
	mScoreBoard.resize(1);
	mScoreBoard[0] = 0.5;
	mAccScore.resize(mNumChars);
	mAccScore.setZero();


}

void 
Environment::
resetCharacterPositions()
{
	std::vector<Eigen::Vector2d> charPositions;
	charPositions.push_back(Eigen::Vector2d(-1.0, 0.5));
	// charPositions.push_back(Eigen::Vector2d(-1.0, -0.5));
	charPositions.push_back(Eigen::Vector2d(1.0, 0.5));
	// charPositions.push_back(Eigen::Vector2d(1.0, -0.5));

	for(int i=0;i<2;i++)
	{
		mCharacters[i]->getSkeleton()->setPositions(charPositions[i]);
		mCharacters[i]->getSkeleton()->setVelocities(Eigen::Vector2d(0.0, 0.0));
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
	// setSkelCollidable(wallSkel);
	mWorld->addSkeleton(wallSkel);
}

void 
Environment::
initFloor()
{
	floorSkel = SkelHelper::makeFloor();
	mWorld->addSkeleton(floorSkel);
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
	int colWall = getCollidingWall(skel, radius);
	switch(colWall)
	{
		case 0:
		if(me*skel->getVelocity(0)>0)
			skel->setVelocity(0, -me*skel->getVelocity(0));
		break;
		case 1:
		if(me*skel->getVelocity(0)<0)
			skel->setVelocity(0, -me*skel->getVelocity(0));
		break;
		case 2:
		if(me*skel->getVelocity(1)>0)
			skel->setVelocity(1, -me*skel->getVelocity(1));
		break;
		case 3:
		if(me*skel->getVelocity(1)<0)
			skel->setVelocity(1, -me*skel->getVelocity(1));
		break;
		default: break;
	}
}

void
Environment::
handleBallContact(int index, double radius, double me)
{
	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	// Eigen::VectorXd ballVel = ballSkel->getVelocities();
	double ballDistance = (ballSkel->getPositions() - skel->getPositions()).norm();

	if(ballDistance<0.15+radius)
	{
		for(int i=0;i<mNumChars;i++)
		{
			if(ballDistance > (ballSkel->getPositions() - mCharacters[i]->getSkeleton()->getPositions()).norm())
				return;
		}
		Eigen::VectorXd relativeVel = skel->getVelocities() - ballSkel->getVelocities();
		ballSkel->setVelocities(skel->getVelocities() + me * relativeVel);
		mKicked[index] = 1;
	}

}

void 
Environment::
boundBallVelocitiy(double maxVel)
{
	Eigen::VectorXd ballVel = ballSkel->getVelocities();
	for(int i=0;i<ballVel.size();i++)
	{
		if(abs(ballVel[i])>maxVel)
		{
			ballVel[i] = ballVel[i]/abs(ballVel[i])*9.0;
		}
	}
	ballSkel->setVelocities(ballVel);	
}


void
Environment::
step()
{
	// std::cout<<ballSkel->getCOM().transpose()<<std::endl;
	// std::cout<<mCharacters[0]->getSkeleton()->getVelocities().transpose()<<std::endl;
	for(int i=0;i<mCharacters.size();i++)
	{
		handleWallContact(mCharacters[i]->getSkeleton(), 0.08, 0.5);
		// handleBallContact(mCharacters[i]->getSkeleton(), 0.08, 1.3);
		handleBallContact(i, 0.08, 1.3);
		// handleContact(mCharacters[i]->getSkeleton(), 0.08, 0.5);
	}
	// handleContact(ballSkel, 0.08);
	handleWallContact(ballSkel, 0.08, 0.8);

	boundBallVelocitiy(9.0);


	// ballSkel->setVelocities(ballSkel->getVelocities()*0.90);
	Eigen::VectorXd ballForce = ballSkel->getForces();
	// cout<<ballForce.transpose()<<endl;
	// cout<<ballSkel->getVelocities().transpose()<<endl;
	// cout<<endl;
	ballForce -= 2*ballSkel->getVelocities();
	ballSkel->setForces(ballForce);
	// for(int i=0;i<mCharacters.size();i++)
	// {
	// 	handleContact(mCharacters[i]->getSkeleton(), 0.08, 0.5);
	// }
	// handleContact(ballSkel, 0.08);
	

	// check the reward for debug
	updateScoreBoard();


	getRewards();
	mWorld->step();
	
	// std::cout<<mCharacters[0]->getSkeleton()->getVelocities().transpose()<<std::endl;
	// std::cout<<endl;
}

// The one carring ball. -1 : no one, 0~3 : player
int
Environment::
getDribblerIndex()
{
	return curDribblerIndex;
}

Eigen::VectorXd
Environment::
getState(int index)
{
	// Character's state
	Eigen::Vector2d p,v;
	Character2D* character = mCharacters[index];
	p = character->getSkeleton()->getPositions();
	v = character->getSkeleton()->getVelocities();
	// int isDribbler = (getDribblerIndex()==index) ? 1 : 0;


	// Ball's state
	Eigen::Vector2d ballP, ballV;
	ballP = ballSkel->getPositions() - p;
	ballV = ballSkel->getVelocities() - v;

	// Observation
	std::string teamName = character->getName().substr(0,1);
	
	// Do not count for cur Character
	double distance[mCharacters.size()-1];
	int distanceIndex[mCharacters.size()-1];
	int count =0;

	// We will get the other's position & velocities in sorted form.
	for(int i=0;i<mCharacters.size();i++)
	{
		if(i != index)
		{
			distanceIndex[count] = i;
			Eigen::Vector2d curP = mCharacters[i]->getSkeleton()->getPositions();
			distance[count] = (curP-p).norm();
			count++;
		}
	}
	count = 0;

	Eigen::VectorXd ballPossession(1);
	if(index == 0)
	{
		ballPossession[0] = mScoreBoard[0];
	}
	else
	{
		ballPossession[0] = 1-mScoreBoard[0];
	}
	// Selection Sort
	// double min = DBL_MAX;
	// int minIndex = 0;
	// for(int i=0;i<mCharacters.size()-1;i++)
	// {
	// 	for(int j=i;j<mCharacters.size();j++)
	// 	{
	// 		if(distance[j]<min)
	// 		{
	// 			min = distance[j];
	// 			minIndex = j;
	// 		}
	// 	}
	// 	distance[minIndex] = distance[i];
	// 	distanceIndex[minIndex] = distanceIndex[i];

	// 	distance[i] = min;
	// 	distanceIndex[i] = minIndex;

	// 	min = distance[i+1];
	// 	minIndex = i+1;
	// }

	// Get the number of current team and opponent team
	int numCurTeam=0;
	int numOppTeam=0;
	for(int i=0;i<mCharacters.size();i++)
	{
		if(teamName == mCharacters[i]->getName().substr(0,1))
			numCurTeam++;
		else
			numOppTeam++;
	}

	// Fill in the other agent's relational state in team+distance order
	// 5 = positionDof(2) + velocityDof(2)
	Eigen::VectorXd otherS((numCurTeam-1)*4 + numOppTeam*4);

	for(int i=0;i<mCharacters.size()-1;i++)
	{
		if(mCharacters[distanceIndex[i]]->getName() == teamName)
		{
			SkeletonPtr skel = mCharacters[distanceIndex[i]]->getSkeleton();
			otherS.segment(count*4,2) = skel->getPositions() - p;
			otherS.segment(count*4+2,2) = skel->getVelocities() - v;
			// otherS.segment(count*5+4,1)[0] = (getDribblerIndex()==index) ? 1 : 0;
			count++;
		}
	}
	for(int i=0;i<mCharacters.size()-1;i++)
	{
		if(mCharacters[distanceIndex[i]]->getName() != teamName)
		{
			SkeletonPtr skel = mCharacters[distanceIndex[i]]->getSkeleton();
			otherS.segment(count*4,2) = skel->getPositions() - p;
			otherS.segment(count*4+2,2) = skel->getVelocities() - v;
			// otherS.segment(count*5+4,1)[0] = (getDribblerIndex()==index) ? 1 : 0;
			count++;
		}
	}
	// otherS.resize(0);
	
	// Fill in the goal basket's relational position
	Eigen::VectorXd goalpostPositions(4);
	if(teamName == mGoalposts[0].first)
	{
		goalpostPositions.segment(0,2) = mGoalposts[0].second.segment(0,2) - p;
		goalpostPositions.segment(2,2) = mGoalposts[1].second.segment(0,2) - p;
	}
	else
	{
		goalpostPositions.segment(0,2) = mGoalposts[1].second.segment(0,2) - p;
		goalpostPositions.segment(2,2) = mGoalposts[0].second.segment(0,2) - p;
	}

	// Put these arguments in a single vector

	Eigen::VectorXd s(p.rows() + v.rows() + ballP.rows() + ballV.rows() +
	ballPossession.rows() + otherS.rows() + goalpostPositions.rows());


	// Eigen::VectorXd s(p.rows() + v.rows() + ballP.rows() + ballV.rows() + 1);

	// s.segment(0, p.rows()) = p;
	// s.segment(p.rows(), v.rows()) = v;
	// s.segment(p.rows() + v.rows(), 1)[0] = isDribbler;
	// s.segment(p.rows() + v.rows() + 1, otherS.size()) = otherS;
	// s.segment(p.rows() + v.rows() + 1 + otherS.size(), 4) = goalpostPositions;
	s<<p,v,ballP,ballV,ballPossession,otherS,goalpostPositions;
	return s;
}

Eigen::VectorXd
Environment::
updateScoreBoard(std::string teamName)
{
	if(mKicked[0] == 1)
	{
		mScoreBoard[0] = 1;
		mKicked[0] = 0;
	}
	if(mKicked[1] == 1)
	{
		mScoreBoard[0] = 0;
		mKicked[1] = 0;
	}


	// if(teamName == "A")
	// {
		
	// }
	// else if (teamName == "B")
	// {
	// 	if(mKicked[1] == 1)
	// 		mScoreBoard[0] = 1;
	// 	if(mKicked[0] == 1)
	// 		mScoreBoard[0] = 0;
	// }

	return mScoreBoard;
}

double
Environment::
getReward(int index)
{

	double reward = 0;

	double ballRadius = 0.1;
	double goalpostSize = 1.5;


	Eigen::Vector2d ballPosition = ballSkel->getPositions();
	Eigen::Vector2d widthVector = Eigen::Vector2d::UnitX();
	Eigen::Vector2d heightVector = Eigen::Vector2d::UnitY();
	// if((mCharacters[index]->getSkeleton()->getPositions() - ballPosition).norm() < 0.08 + )
	// {

	// }

	if(index == 0)
	{
		reward += 0.1 * mScoreBoard[0];
	}
	else
	{
		reward += 0.1 * (1-mScoreBoard[0]);
	}


	// reward += 1.0 * mKicked[index];
	


	// reward += 0.1 * exp(-pow((mCharacters[index]->getSkeleton()->getPositions() - ballPosition).norm(),2.0));


	/*
	if(index<2)
	{

		// reward += exp(-(mGoalposts[1].second.segment(0,2) - ballPosition).norm()/3.0);
		reward += 0.01 * exp(-pow((mCharacters[index]->getSkeleton()->getPositions() - ballPosition).norm(),2.0));
		// std::cout<<widthVector.normalized().dot(ballPosition)-ballRadius<<" ";
		// std::cout<< mGoalposts[0].second.x()<<endl;
		// cout<<(mGoalposts[1].second.segment(0,2) - ballPosition).transpose()<<endl;
		if(widthVector.normalized().dot(ballPosition)-ballRadius <= mGoalposts[0].second.x())
		{
			if(abs(heightVector.normalized().dot(ballPosition)) < goalpostSize/2.0)
			{
				// mIsTerminalState = true;
				// std::cout<<"Blue Team GOALL!!"<<std::endl;
				// reward += -100;
			}
		}
		else if(widthVector.normalized().dot(ballPosition)+ballRadius >= mGoalposts[1].second.x())
		{
			if(abs(heightVector.normalized().dot(ballPosition)) < goalpostSize/2.0)
			{
				// mIsTerminalState = true;
				// std::cout<<"Red Team GOALL!!"<<std::endl;
				// reward += 100;
			}
		}

	}
	else
	{

		// reward = exp(-(mGoalposts[0].second.segment(0,2) - ballPosition).norm()/3.0);
		reward += 0.01 * exp(-pow((mCharacters[index]->getSkeleton()->getPositions() - ballPosition).norm(),2.0));

		if(widthVector.normalized().dot(ballPosition)-ballRadius <= mGoalposts[0].second.x())
		{
			if(abs(heightVector.normalized().dot(ballPosition)) < goalpostSize/2.0)
			{
				// mIsTerminalState = true;
				// std::cout<<"Blue Team GOALL!!"<<std::endl;
				// reward += +100;
			}
		}
		else if(widthVector.normalized().dot(ballPosition)+ballRadius >= mGoalposts[1].second.x())
		{
			if(abs(heightVector.normalized().dot(ballPosition)) < goalpostSize/2.0)
			{
				// mIsTerminalState = true;
				// std::cout<<"Red Team GOALL!!"<<std::endl;
				// reward += -100;
			}
		}
	}
	*/
	return reward;
}

std::vector<double> 
Environment::
getRewards()
{
	std::vector<double> rewards;
	// for(int i=0;i<mCharacters.size();i++)
	for(int i=0;i<mNumChars;i++)
	{
		rewards.push_back(getReward(i));
		mAccScore[i] += rewards[i];
	}

	// cout.setf(ios::fixed);
	// cout.precision(3);
	// cout<<"\r"<<rewards[0]<<" "<<rewards[1];
	// cout.unsetf(ios::fixed);
	return rewards;
}

void 
Environment::
setAction(int index, const Eigen::VectorXd& a)
{
	mTimeElapsed += 1.0 / (double)mControlHz;
	double maxVel = 3.0;
	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	Eigen::VectorXd vel = skel->getVelocities();

	// controll the ball if the char is close to ball.
	Eigen::VectorXd controlForceToBall(2);

	vel += a.segment(0, vel.size());
	// cout<<vel.transpose()<<endl;

	for(int i=0;i<vel.size();i++)
	{
		if(abs(vel[i])>maxVel)
		{
			vel[i] /= abs(vel[i]);
			vel[i] *= maxVel;
		}
	}
	skel->setVelocities(vel);

	// double controlPower;
	// double relativeBallPosition 
	// = (skel->getCOM().segment(0,2) - ballSkel->getCOM().segment(0,2)).norm();

	// controlPower = exp(-pow(relativeBallPosition,2)/1.0);



	
	// if(index==0)
	// 	cout<<controlPower<<endl;
	// if(index==1)
	// 	cout<<ballSkel->getForces().transpose()<<endl;



	// ballSkel->setForces(ballSkel->getForces()+controlForceToBall*controlPower*100.0);



	// if(index==1)
	// 	cout<<ballSkel->getForces().transpose()<<endl;
	// cout<<endl;;
	// cout<<controlForceToBall.transpose()<<endl;
	// vel+= a/10.0;

}
// void
// Environment::
// setActions(std::vector<Eigen::VectorXd> as)
// {
// 	for(int i=0;i<mCharacters.size();i++)
// 	{
// 		setAction(i, as[i]);
// 	}
// }

bool
Environment::
isTerminalState()
{
	if(mTimeElapsed>15.0)
		mIsTerminalState = true;
	return mIsTerminalState;
}

void
Environment::
reset()
{
	Eigen::VectorXd ballPosition = ballSkel->getPositions();
	ballPosition[0] = 6.0 * (rand()/(double)RAND_MAX ) - 6.0/2.0;
	ballPosition[1] = 4.5 * (rand()/(double)RAND_MAX ) - 4.5/2.0;
	ballSkel->setPositions(ballPosition);
	Eigen::VectorXd ballVel = ballSkel->getVelocities();
	ballVel[0] = 10.0 * (rand()/(double)RAND_MAX ) - 5.0;
	ballVel[1] = 10.0 * (rand()/(double)RAND_MAX ) - 5.0;
	ballSkel->setVelocities(ballVel);

	for(int i=0;i<mNumChars;i++)
	{
		SkeletonPtr skel = mCharacters[i]->getSkeleton();
		Eigen::VectorXd skelPosition = skel->getPositions();
		skelPosition[0] = 6.0 * (rand()/(double)RAND_MAX ) - 6.0/2.0;
		skelPosition[1] = 4.5 * (rand()/(double)RAND_MAX ) - 4.5/2.0;
		skel->setPositions(skelPosition);
		Eigen::VectorXd skelVel = skel->getVelocities();
		skelVel[0] = 3.0 * (rand()/(double)RAND_MAX ) - 1.5;
		skelVel[1] = 3.0 * (rand()/(double)RAND_MAX ) - 1.5;
		skel->setVelocities(skelVel);
	}


	mIsTerminalState = false;
	mTimeElapsed = 0;

	mScoreBoard[0] = 0.5;

	mAccScore.setZero();

	// resetCharacterPositions();
}




// 0: positive x / 1: negative x / 2 : positive y / 3 : negative y
int
Environment::
getCollidingWall(SkeletonPtr skel, double radius)
{
	// double charRadius = 0.07;
	double groundWidth = 4.0;
	double groundHeight = 3.0;

	Eigen::Vector2d centerVector = Eigen::Vector2d(skel->getCOM().x(), skel->getCOM().y());
	Eigen::Vector2d widthVector = Eigen::Vector2d::UnitX();
	Eigen::Vector2d heightVector = Eigen::Vector2d::UnitY();

	if(centerVector.dot(widthVector) >= (groundWidth-radius))
		return 0;
	else if(centerVector.dot(widthVector) <= -(groundWidth-radius))
		return 1;
	else if(centerVector.dot(heightVector) >= (groundHeight-radius))
		return 2;
	else if(centerVector.dot(heightVector) <= -(groundHeight-radius))
		return 3;

	return -1;
}