#include "Environment.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include <iostream>
#include <chrono>

using namespace std;
using namespace dart;
using namespace dart::dynamics;

Environment::
Environment(int control_Hz, int simulation_Hz, int numChars)
:mControlHz(control_Hz), mSimulationHz(simulation_Hz), mNumChars(numChars), mWorld(std::make_shared<dart::simulation::World>())
{
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
	for(int i=0;i<2;i++)
	{
		mCharacters.push_back(new Character2D("A_" + to_string(i)));
	}
	for(int i=0;i<2;i++)
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
		mActions.push_back(zeroVel);
	}


}

void 
Environment::
resetCharacterPositions()
{
	std::vector<Eigen::Vector2d> charPositions;
	charPositions.push_back(Eigen::Vector2d(-1.0, 0.5));
	charPositions.push_back(Eigen::Vector2d(-1.0, -0.5));
	charPositions.push_back(Eigen::Vector2d(1.0, 0.5));
	charPositions.push_back(Eigen::Vector2d(1.0, -0.5));

	for(int i=0;i<4;i++)
	{
		mCharacters[i]->getSkeleton()->setPositions(charPositions[i]);
		mCharacters[i]->getSkeleton()->setVelocities(Eigen::Vector2d(0.0, 0.0));
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
	mWorld->addSkeleton(ballSkel);
}

void
Environment::
step()
{
	// std::cout<<ballSkel->getCOM().transpose()<<std::endl;
	// std::cout<<mCharacters[0]->getSkeleton()->getVelocities().transpose()<<std::endl;
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
	int isDribbler = (getDribblerIndex()==index) ? 1 : 0;

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
		}
	}

	// Selection Sort
	double min = DBL_MAX;
	int minIndex = 0;
	for(int i=0;i<mCharacters.size()-1;i++)
	{
		for(int j=i;j<mCharacters.size();j++)
		{
			if(distance[j]<min)
			{
				min = distance[j];
				minIndex = j;
			}
		}
		distance[minIndex] = distance[i];
		distanceIndex[minIndex] = distanceIndex[i];

		distance[i] = min;
		distanceIndex[i] = minIndex;

		min = distance[i+1];
		minIndex = i+1;
	}

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
	// 5 = positionDof(2) + velocityDof(2) + isDribbler
	Eigen::VectorXd otherS((numCurTeam-1)*5 + numOppTeam*5);
	count = 0;
	for(int i=0;i<mCharacters.size()-1;i++)
	{
		if(mCharacters[distanceIndex[i]]->getName() == teamName)
		{
			SkeletonPtr skel = mCharacters[distanceIndex[i]]->getSkeleton();
			otherS.segment(count*5,2) = skel->getPositions() - p;
			otherS.segment(count*5+2,2) = skel->getVelocities() - v;
			otherS.segment(count*5+4,1)[0] = (getDribblerIndex()==index) ? 1 : 0;
			count++;
		}
	}
	for(int i=0;i<mCharacters.size()-1;i++)
	{
		if(mCharacters[distanceIndex[i]]->getName() != teamName)
		{
			SkeletonPtr skel = mCharacters[distanceIndex[i]]->getSkeleton();
			otherS.segment(count*5,2) = skel->getPositions() - p;
			otherS.segment(count*5+2,2) = skel->getVelocities() - v;
			otherS.segment(count*5+4,1)[0] = (getDribblerIndex()==index) ? 1 : 0;
			count++;
		}
	}

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
	Eigen::VectorXd s(p.rows() + v.rows() + 1 + otherS.size() + 4);
	// s.segment(0, p.rows()) = p;
	// s.segment(p.rows(), v.rows()) = v;
	// s.segment(p.rows() + v.rows(), 1)[0] = isDribbler;
	// s.segment(p.rows() + v.rows() + 1, otherS.size()) = otherS;
	// s.segment(p.rows() + v.rows() + 1 + otherS.size(), 4) = goalpostPositions;
	s<<p,v,isDribbler,otherS,goalpostPositions;

	return s;
}

double
Environment::
getReward(int index)
{
	if(index<2)
	{
		Eigen::Vector3d ballPosition = ballSkel->getCOM();
		if(ballPosition.x()<mGoalposts[0].second.x())
		{
			std::cout<<"GOAL!!"<<std::endl;
			return -1;
		}
	}
	else
	{
		Eigen::Vector3d ballPosition = ballSkel->getCOM();
		if(ballPosition.x()>mGoalposts[1].second.x())
		{
			std::cout<<"GOAL!!"<<std::endl;
			return 1;
		}
	}
	return 0;
}

std::vector<double> 
Environment::
getRewards()
{
	std::vector<double> rewards;
	for(int i=0;i<mCharacters.size();i++)
	{
		rewards.push_back(getReward(i));
	}

	return rewards;
}

void 
Environment::
setAction(int index, const Eigen::VectorXd& a)
{
	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	Eigen::VectorXd vel = skel->getVelocities();
	vel+= a;
	skel->setVelocities(vel);
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
	for(int i=0;i<2;i++)
	{
		Eigen::Vector3d ballPosition = ballSkel->getCOM();
		if(ballPosition.x()<mGoalposts[0].second.x())
			return true;
		else if(ballPosition.x()>mGoalposts[1].second.x())
			return true;

	}
	return false;
}

void
Environment::
reset()
{
	Eigen::VectorXd ballPosition = ballSkel->getPositions();
	ballPosition.setZero();
	ballSkel->setPositions(ballPosition);

	resetCharacterPositions();
}