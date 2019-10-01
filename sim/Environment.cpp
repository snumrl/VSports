#include "Environment.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include "dart/external/lodepng/lodepng.h"
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
	initPrevTargetPositions();
	getNumState();
	// initGoalState();
	mWorld->getConstraintSolver()->removeAllConstraints();
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

		mAction.resize(zeroVel.rows()+touch.rows());
		mAction << zeroVel, touch;
		mActions.push_back(mAction);
	}
	// mTouch.resize(mNumChars);

	mKicked.resize(2);
	mKicked.setZero();
	mScoreBoard.resize(1);
	mScoreBoard[0] = 0.5;
	mPrevScoreBoard.resize(1);
	mPrevScoreBoard[0] = mScoreBoard[0];
	mAccScore.resize(mNumChars);
	mAccScore.setZero();
	mStates.resize(mNumChars);
	mPrevStates.resize(mNumChars);
	mForces.resize(mNumChars);
	mPrevTargetPositions.resize(mNumChars);

	mMapStates.resize(mNumChars);

	for(int i=0;i<mNumChars;i++)
	{
		mMapStates[i] = new MapState(4);
		goalRewardPaid.push_back(false);
	}

	mSubGoalRewards.resize(mNumChars);

}

void
Environment::
initPrevTargetPositions()
{
	for(int i=0;i<mNumChars;i++)
	{
		mPrevTargetPositions[i] = ballSkel->getPositions();
	}
}
void 
Environment::
resetCharacterPositions()
{
	if(mNumChars == 4)
	{
		std::vector<Eigen::Vector2d> charPositions;
		charPositions.push_back(Eigen::Vector2d(-1.0, 0.5));
		charPositions.push_back(Eigen::Vector2d(-1.0, -0.5));
		charPositions.push_back(Eigen::Vector2d(1.0, 0.5));
		charPositions.push_back(Eigen::Vector2d(1.0, -0.5));

		for(int i=0;i<mNumChars;i++)
		{
			mCharacters[i]->getSkeleton()->setPositions(charPositions[i]);
			mCharacters[i]->getSkeleton()->setVelocities(Eigen::Vector2d(0.0, 0.0));
		}
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
	// for(int i=0;i<collidingWalls.size();i++)
	// {
	// 	cout<<collidingWalls[i]<<" ";
	// }
	// cout<<endl;
	// cout<<colWall<<endl;
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
				// cout<<"1, here?"<<endl;
				skel->setVelocity(0, -me*skel->getVelocity(0));
				skel->setForce(0, 0);
			break;
			case 2:
			if(me*skel->getVelocity(1)>0)
				// cout<<"2, here?"<<endl;
				// cout<<skel->getVelocities().transpose()<<endl;
				skel->setVelocity(1, -me*skel->getVelocity(1));
				// cout<<skel->getVelocities().transpose()<<endl;
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
	// Eigen::VectorXd ballVel = ballSkel->getVelocities();
	double ballDistance = (ballSkel->getPositions() - skel->getPositions()).norm();

	double kickPower = mActions[index][2];

	// if(kickPower>=1)
	// 	kickPower = 1;

	if(kickPower >= 0)
	{
		// kickPower = 1.0/(exp(-kickPower)+1);
		// cout<<"Kicked!"<<endl;
		kickPower = 1.0;
		ballSkel->setVelocities(skel->getVelocities()*(2.0+me)*(1.0*kickPower));
		if(mCharacters[index]->getTeamName() == "A")
			mKicked[0] = 1;
		else
			mKicked[1] = 1;
	}

}

void
Environment::
handlePlayerContact(int index1, int index2, double me)
{
	SkeletonPtr skel1 = mCharacters[index1]->getSkeleton();
	SkeletonPtr skel2 = mCharacters[index2]->getSkeleton();

	double playerRadius = 0.25;

	Eigen::Vector2d skel1Positions = skel1->getPositions();
	Eigen::Vector2d skel2Positions = skel2->getPositions();

	if((skel1Positions - skel2Positions).norm() < playerRadius)
	{
		Eigen::Vector2d skel1Velocities = skel1->getVelocities();
		Eigen::Vector2d skel2Velocities = skel2->getVelocities();

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

		skel1->setVelocities(skel1NewVelocities);
		skel2->setVelocities(skel2NewVelocities);
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
			ballVel[i] = ballVel[i]/abs(ballVel[i])*9.0;
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

	// handlePlayerContacts(0.5);

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

	boundBallVelocitiy(9.0);
	dampBallVelocitiy(2.0);

	updateScoreBoard();

	// cout<<mCharacters[0]->getSkeleton()->getVelocities().transpose()<<endl;

	mWorld->step();
	// cout<<"------------------"<<endl;
	// cout<<mCharacters[0]->getSkeleton()->getVelocities().transpose()<<endl;

}


void
Environment::
stepAtOnce()
{
	int sim_per_control = this->getSimulationHz()/this->getControlHz();

	// for(int i=0;i<mPrevStates.size();i++)
	// {
	// 	mPrevStates[i] = mStates[i];
	// }
	mPrevStates = mStates;

	for(int i=0;i<sim_per_control;i++)
	{
		this->step();
	}
	// updateState();
	// getRewards();
	mPrevScoreBoard = mScoreBoard;
}

void
Environment::
updateState()
{
	for(int i=0;i<mNumChars;i++)
	{
		getState(i);
	}
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

	Eigen::VectorXd distanceWall(4);

	// Ball's state
	Eigen::Vector2d ballP, ballV;
	ballP = ballSkel->getPositions();
	ballV = ballSkel->getVelocities();

	Eigen::Vector2d relativeBallP, relativeBallV;
	relativeBallP = ballP - p;
	relativeBallV = ballV - v;

	std::string teamName = character->getName().substr(0,1);
	if(teamName == mGoalposts[0].first)
		distanceWall << 4-ballP[0], 4+ballP[0], 3-ballP[1], 3+ballP[1];
	else	
		distanceWall << 4+ballP[0], 4-ballP[0], 3+ballP[1], 3-ballP[1];

	Eigen::VectorXd ballPossession(1);
	if(index == 0)
	{
		ballPossession[0] = mScoreBoard[0];
	}
	else
	{
		ballPossession[0] = 1-mScoreBoard[0];
	}

	Eigen::VectorXd kickable(1);
	if(relativeBallP.norm()<0.15+0.08)
	{
		kickable[0] = 1;
	}
	else
	{
		kickable[0] = 0;
	}
	
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

	// Observation
	
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
			count++;
		}
	}

	Eigen::VectorXd goalpostPositions(8);
	if(teamName == mGoalposts[0].first)
	{
		goalpostPositions.segment(0,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - ballP;
		goalpostPositions.segment(2,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - ballP;
		goalpostPositions.segment(4,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - ballP;
		goalpostPositions.segment(6,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - ballP;
	}
	else
	{
		goalpostPositions.segment(0,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - ballP;
		goalpostPositions.segment(2,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - ballP;
		goalpostPositions.segment(4,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - ballP;
		goalpostPositions.segment(6,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - ballP;
	}


	double reviseStateByTeam = -1;
	if(teamName == mGoalposts[0].first)
	{
		// cout<<index<<endl;
		reviseStateByTeam = 1;
	}

	// Put these arguments in a single vector
	bool useMap = false;
	std::vector<float> mapState;
	if(useMap)
	{
		setStateMinimap(index);
		mapState = mMapStates[index]->getVectorizedValue();
	}

	Eigen::VectorXd vecState(p.rows() + v.rows() + relativeBallP.rows() + relativeBallV.rows() +
	ballPossession.rows() + kickable.rows() + goalpostPositions.rows());


	vecState<<p,v,relativeBallP,relativeBallV,ballPossession,kickable,goalpostPositions;


	Eigen::VectorXd simpleGoalpostPositions(8);
	if(teamName == mGoalposts[0].first)
	{
		simpleGoalpostPositions.segment(0,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - ballP;
		simpleGoalpostPositions.segment(2,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - ballP;
		simpleGoalpostPositions.segment(4,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - p;
		simpleGoalpostPositions.segment(6,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - p;
	}

	Eigen::VectorXd distanceWall_char(4);
	Eigen::VectorXd distanceWall_ball(4);

	// Ball's state

	// std::string teamName = character->getName().substr(0,1);
	if(teamName == mGoalposts[0].first)
		distanceWall_char << 4-p[0], 4+p[0], 3-p[1], 3+p[1];
	else	
		distanceWall_char << 4+p[0], 4-p[0], 3+p[1], 3-p[1];

	if(teamName == mGoalposts[0].first)
		distanceWall_ball << 4-ballP[0], 4+ballP[0], 3-ballP[1], 3+ballP[1];
	else	
		distanceWall_ball << 4+ballP[0], 4-ballP[0], 3+ballP[1], 3-ballP[1];


	Eigen::VectorXd state;

	// 	state.resize(p.rows() + v.rows() + relativeBallP.rows() + relativeBallV.rows() +
	// 		ballPossession.rows() + kickable.rows() + otherS.rows() + distanceWall.rows() + goalpostPositions.rows());

	state.resize(p.rows() + v.rows() + relativeBallP.rows() + relativeBallV.rows() + kickable.rows() + simpleGoalpostPositions.rows());
		// +distanceWall_char.rows() + distanceWall_ball.rows());
	

	count = 0;

	for(int i=0;i<p.rows();i++)
	{
		state[count++] = reviseStateByTeam * p[i];
	}
	for(int i=0;i<v.rows();i++)
	{
		state[count++] = reviseStateByTeam * v[i];
	}
	for(int i=0;i<relativeBallP.rows();i++)
	{
		state[count++] = reviseStateByTeam * relativeBallP[i];
	}
	for(int i=0;i<relativeBallV.rows();i++)
	{
		state[count++] = reviseStateByTeam *  relativeBallV[i];
	}

	// for(int i=0;i<ballPossession.rows();i++)
	// {
	// 	state[count++] = ballPossession[i];
	// }
	for(int i=0;i<kickable.rows();i++)
	{
		state[count++] = kickable[i];
	}


	// else
	// {
	// 	for(int i=0;i<otherS.rows();i++)
	// 	{
	// 		state[count++] = reviseStateByTeam * otherS[i];
	// 	}
	// 	for(int i=0;i<distanceWall.rows();i++)
	// 	{
	// 		state[count++] = distanceWall[i];
	// 	}
	// }

	for(int i=0;i<simpleGoalpostPositions.rows();i++)
	{
		state[count++] = reviseStateByTeam * simpleGoalpostPositions[i];
	}

	// for(int i=0;i<distanceWall_char.rows();i++)
	// {
	// 	state[count++] = distanceWall_char[i];
	// }
	// for(int i=0;i<distanceWall_ball.rows();i++)
	// {
	// 	state[count++] = distanceWall_ball[i];
	// }

	mStates[index] = state;
	return normalizeNNState(state);
}

Eigen::VectorXd
Environment::
getSchedulerState(int index)
{
	Eigen::VectorXd combinedState;
	Eigen::VectorXd state = getState(index) ;
	Eigen::VectorXd targetPosition(2);
	double targetShooting;
	combinedState.resize(state.size() + targetPosition.size());

	Eigen::VectorXd p = mStates[index].segment(_ID_P,2);
	Eigen::VectorXd ballP = p + mStates[index].segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = ballP + (mStates[index].segment(_ID_GOALPOST_P, 2) + mStates[index].segment(_ID_GOALPOST_P+2, 2))/2.0;
	// targetShooting = 0.5;

	Eigen::VectorXd tempVec = (centerOfGoalpost - p).normalized().dot((ballP - p)) *  (centerOfGoalpost - p).normalized();
	double ballToLine = ((ballP - p) - tempVec).norm();
	Eigen::VectorXd targetP = centerOfGoalpost + (ballP - centerOfGoalpost) + 0.3 * (ballP - centerOfGoalpost).normalized();
	// targetP = centerOF
	// targetPosition = targetP - p;
	if (ballToLine >= 0.30)
	{
		// if(index == 0)
		// 	cout<<0<<endl;
		targetPosition = targetP - p;
	}
	else if( (ballP-centerOfGoalpost).norm()-0.20 >(p-centerOfGoalpost).norm()  )
	{
		// cout<<state[_ID_KICKABLE]<<endl;
		// if(index == 0)
		// 	cout<<1<<endl;
		targetPosition = targetP - p;
	}
	else
	{
		// cout<<1<<endl;
		// if(index == 0)
		// 	cout<<2<<endl;
		targetP = centerOfGoalpost + (ballP - centerOfGoalpost) - 0.3 * (ballP - centerOfGoalpost).normalized();
		targetPosition = targetP - p;
		// double curShooting = mActions[index][2];
		// double targetShooting = 1.0;

		// reward = 1.0 + exp(-(targetP - p).norm()) + exp(-(targetShooting-curShooting));
	}
	// normalize the target position
	// mPrevTargetPositions[index] = targetPosition;
	combinedState << state, targetPosition/4.0;


	// assert(state.size() == mGoalStates[index].size());

	// combinedState.segment(0, state.size()) = state;
	// combinedState.segment(state.size(), mGoalStates[index].size()) = normalizeNNState((mGoalStates[index] - unNormalizeNNState(state)).cwiseProduct(mWGoalStates[index]));

	// if(index == 0)
	// 	cout<<combinedState.segment(state.size() + _ID_BALL_P, 2).norm()<<endl;

	// for(int i=0;i<state.size();i++)
	// {
	// 	combinedState[i] = state[i];
	// }
	// for(int i=0;i<mGoalStates[index].size();i++)
	// {
	// 	combinedState[state.size()+i] = (state[i] - mGoalStates[index][i])*mWGoalStates[index][i];
	// }

	return state;
}

Eigen::VectorXd
Environment::
getLinearActorState(int index)
{
	Eigen::VectorXd linearActorState(mStates[index].size()*2);
	linearActorState.segment(0, mStates[index].size()) = normalizeNNState(mStates[index]);
	linearActorState.segment(mStates[index].size(), mStates[index].size()) = normalizeNNState(mSubgoalStates[index]).cwiseProduct(mWSubgoalStates[index]);
	return linearActorState;
}




std::pair<int, int> getPixelFromPosition(double x, double y, double maxX = 4.0, double maxY = 3.0, 
													int numRows = 84, int numCols = 84)
{
	int pixelX, pixelY;

	int midX = numRows/2;
	int midY = numCols/2;
	if(maxX>=maxY)
	{
		// +0.5 for bounding
		pixelX = numRows/2 * (x/maxX) + midX + 0.5;
		pixelY = midY - (numCols/2 * (y/maxY))*(maxY/maxX) + 0.5;
	}
	else
	{
		pixelX = midX + (numRows/2 * (x/maxX))*(maxX/maxY) + 0.5;
		pixelY = midY - (numCols/2 * (y/maxY)) + 0.5;	
	}
	if(pixelX > numRows)
		pixelX = numRows;
	if (pixelX <= 1)
		pixelX = 1;
	if(pixelY > numCols)
		pixelY = numCols;
	if(pixelY <=1)
		pixelY = 1;
	return std::make_pair(pixelX-1, pixelY-1);
}

void visualizeMinimap(std::vector<double> minimaps)
{
	for(int i=0;i<5;i++)
	{
		std::vector<unsigned char> image;
		image.resize(40*40*4);
		for(int col=0;col<40;col++)
		{
			for(int row=0;row<40;row++)
			{
				image[(col*40+row)*4+0] = minimaps[40*40*i + col*40+row] * 255;
				image[(col*40+row)*4+1] = minimaps[40*40*i + col*40+row] * 255;
				image[(col*40+row)*4+2] = minimaps[40*40*i + col*40+row] * 255;
				image[(col*40+row)*4+3] = 255;
			}
		}

		lodepng::encode(("./minimap_Image"+to_string(i)+".png").c_str(), image, 40, 40);
	}
}

void
Environment::
setStateMinimap(int index)
{
	// std::vector<Eigen::MatrixXd> minimaps;
	int numRows = 84;
	int numCols = 84; 

	/// map for wall
	Eigen::VectorXd minimap;
	minimap.resize(numRows*numCols);
	minimap.setZero();

	for(int i=0;i<numCols;i++)
	{
		for(int j=0;j<numRows;j++)
		{
			minimap[i*numRows+j] = 0.5;
		}	
	}
	for(double i = -4;i<4;i+= 0.05)
	{
		for(double j=-3;j<3;j+=0.05)
		{
			std::pair<int, int> pixel = getPixelFromPosition(i, j);
			minimap[pixel.second*numRows+pixel.first] = 0.0;
		}
	}

	Eigen::VectorXd position;
	std::pair<int, int> pixel;

	// cur agent position

	position = mCharacters[index]->getSkeleton()->getPositions();
	pixel = getPixelFromPosition(position[0], position[1]);
	minimap[pixel.second*numRows+pixel.first] = 1.0;

	// // team position
	// std::string mTeamName = mCharacters[index]->getTeamName();
	// for(int i=0;i<mCharacters.size();i++)
	// {
	// 	if(i != index)
	// 	{
	// 		if(mCharacters[i]->getTeamName()==mTeamName)
	// 		{
	// 			position = mCharacters[i]->getSkeleton()->getPositions();

	// 			pixel = getPixelFromPosition(position[0], position[1]);
	// 			minimap[pixel.second*numRows+pixel.first] = 0.8;
	// 		}
	// 	}

	// }

	// ball position
	position = ballSkel->getPositions();

	pixel = getPixelFromPosition(position[0], position[1]);
	for(int i=-1;i<=1;i++)
	{
		for(int j=-1;j<=1;j++)
		{
			if(mScoreBoard[0] == 1)
				minimap[(pixel.second+j)*numRows+pixel.first+i] = 0.6;
			else if(mScoreBoard[0] == 0)
				minimap[(pixel.second+j)*numRows+pixel.first+i] = 0.4;
			else
				minimap[(pixel.second+j)*numRows+pixel.first+i] = 0.5;

		}
	}
	if(mScoreBoard[0] == 1)
		minimap[pixel.second*numRows+pixel.first] = 0.7;
	else if(mScoreBoard[0] == 0)
		minimap[pixel.second*numRows+pixel.first] = 0.5;
	else
		minimap[pixel.second*numRows+pixel.first] = 0.6;



	// // opponent team position
	// for(int i=0;i<mCharacters.size();i++)
	// {
	// 	if(mCharacters[i]->getTeamName()!=mTeamName)
	// 	{
	// 		Eigen::VectorXd position = mCharacters[i]->getSkeleton()->getPositions();

	// 		std::pair<int, int> pixel = getPixelFromPosition(position[0], position[1]);
	// 		minimap[pixel.second*numRows+pixel.first] = 0.3;
	// 	}
	// }

	// for(int i=0;i<numCols;i++)
	// {
	// 	for(int j=0;j<numRows;j++)
	// 	{
	// 		minimap[i*numRows+j] *= 1.0;
	// 	}
	// }

	mMapStates[index]->setCurState(minimap);

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
	return mScoreBoard;
}

double
Environment::
getReward(int index)
{

	if(index < 1)
		return 5.0 * exp(-pow((mStates[index].segment(_ID_P,2) + mStates[index].segment(_ID_BALL_P,2) - mGoalposts[1].second.segment(0,2)).norm(),2));
	else
		return 5.0 * exp(-pow((mStates[index].segment(_ID_P,2) + mStates[index].segment(_ID_BALL_P,2) - mGoalposts[1].second.segment(0,2)).norm(),2));

		// return 1.0;

	// return 0.0;

	double reward = 0;

	double vanishingCount = 100.0;

	double clipedIterations;

	clipedIterations = mNumIterations;
	if(clipedIterations > 100.0)
		clipedIterations = 100.0;

	double ballRadius = 0.1;
	double goalpostSize = 1.5;


	Eigen::Vector2d ballPosition = ballSkel->getPositions();
	Eigen::Vector2d widthVector = Eigen::Vector2d::UnitX();
	Eigen::Vector2d heightVector = Eigen::Vector2d::UnitY();


	double mDistanceBall = mStates[index].segment(_ID_BALL_P,2).norm();

	if(mNumIterations < 50)
		reward +=(50 - mNumIterations)/50.0*0.2 * exp(-pow(mDistanceBall,2));

	double rewardScale = 1/30.0;
	


	// // Goal Reward
	// if(index<1)
	// {
	// 	if(widthVector.dot(ballPosition)-ballRadius <= mGoalposts[0].second.x())
	// 	{
	// 		if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
	// 		{
	// 			// std::cout<<"Blue Team GOALL!!"<<std::endl;
	// 			if(!goalRewardPaid[index])
	// 				reward += -160 * rewardScale * (clipedIterations/vanishingCount);
	// 			goalRewardPaid[index] = true;
	// 			mIsTerminalState = true;
	// 		}
	// 	}
	// 	else if(widthVector.dot(ballPosition)+ballRadius >= mGoalposts[1].second.x())
	// 	{
	// 		if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
	// 		{
	// 			// std::cout<<"Red Team GOALL!!"<<std::endl;
	// 			if(!goalRewardPaid[index])
	// 				reward += 160 * rewardScale * (clipedIterations/vanishingCount);
	// 			goalRewardPaid[index] = true;
	// 			mIsTerminalState = true;
	// 		}
	// 	}

	// }
	// else
	// {

	// 	if(widthVector.dot(ballPosition)-ballRadius <= mGoalposts[0].second.x())
	// 	{
	// 		if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
	// 		{
	// 			// std::cout<<"Blue Team GOALL!!"<<std::endl;
	// 			if(!goalRewardPaid[index])
	// 				reward += 160 * rewardScale * (clipedIterations/vanishingCount);
	// 			goalRewardPaid[index] = true;
	// 			mIsTerminalState = true;
	// 		}
	// 	}
	// 	else if(widthVector.dot(ballPosition)+ballRadius >= mGoalposts[1].second.x())
	// 	{
	// 		if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
	// 		{
	// 			// std::cout<<"Red Team GOALL!!"<<std::endl;
	// 			if(!goalRewardPaid[index])
	// 				reward += -160 * rewardScale * (clipedIterations/vanishingCount);
	// 			goalRewardPaid[index] = true;
	// 			mIsTerminalState = true;
	// 		}
	// 	}
	// }

	return reward;
}


// check if the next state is closer to the subgoal state or
// check the action is facing the subgoal state
// We hope we made the subgoal as linear(greedy) part for complex task
double
Environment::
getLinearActorReward(int index)
{
	double reward = 0;

	Eigen::VectorXd prevSubgoalDistance = mSubgoalStates[index].cwiseProduct(mWSubgoalStates[index]);
	// cout<<prevSubgoalDistance.size()<<endl;
	Eigen::VectorXd curSubgoalDistance = (mPrevStates[index] + mSubgoalStates[index] - mStates[index]).cwiseProduct(mWSubgoalStates[index]);
	// cout<<mPrevStates[index].size()<<endl;

	// cout<<mWSubgoalStates[index].transpose()<<endl;
	reward = prevSubgoalDistance.norm() - curSubgoalDistance.norm();

	// reward = 0.1 * exp(0.01 * reward);
	reward *= 0.1;
	mSubGoalRewards[index] = 1.0 * reward;
	// cout<<reward<<endl;

	return reward;
}

// double
// Environment::
// getSchedulerReward(int index)
// {
// 	double goalReward = 0.0;
// 	Eigen::VectorXd weightedDistanceToGoal;
// 	weightedDistanceToGoal = (mGoalStates[index]-(mSubgoalStates[index] + mPrevStates[index])).cwiseProduct(mWGoalStates[index]);
	
// 	// cout<<mGoalStates[index].segment(_ID_BALL_P, 2).transpose()<<endl;
// 	// cout<<mSubgoalStates[index].segment(_ID_BALL_P, 2).transpose()<<endl;
// 	// cout<<mWGoalStates[index].segment(_ID_BALL_P, 2).transpose()<<endl;
// 	// cout<<weightedDistanceToGoal.segment(_ID_BALL_P,2).transpose()<<endl;
// 	// cout<<endl;
// 	if(weightedDistanceToGoal.norm() < 0.53)
// 	{
// 		// cout<<weightedDistanceToGoal.norm()<<endl;
// 		goalReward = 4.0;
// 		// cout<<"Player "<<index<<" reached goal state! "<<weightedDistanceToGoal.norm()<<endl;
// 		// mIsTerminalState = true;
// 	}

// 	// check the state is properly linearly reaching the subgoal
// 	double subgoalReachableReward = 0.0;

// 	// subgoalReachableReward = 1.0 *mSubGoalRewards[index];

// 	return goalReward + subgoalReachableReward;
// }



double
Environment::
getSchedulerReward(int index)
{
	double reward = 0.0;
	Eigen::VectorXd p = mStates[index].segment(_ID_P,2);
	Eigen::VectorXd ballP = p + mStates[index].segment(_ID_BALL_P,2);
	Eigen::VectorXd centerOfGoalpost = ballP + (mStates[index].segment(_ID_GOALPOST_P, 2) + mStates[index].segment(_ID_GOALPOST_P+2, 2))/2.0;

	Eigen::VectorXd tempVec = (centerOfGoalpost - p).normalized().dot((ballP - p)) *  (centerOfGoalpost - p).normalized();
	double ballToLine = ((ballP - p) - tempVec).norm();
	Eigen::VectorXd targetP = centerOfGoalpost + (ballP - centerOfGoalpost) + 0.3 * (ballP - centerOfGoalpost).normalized();
	// reward = exp(-2.0 * (targetP - p).squaredNorm()) + 0.2 * exp(-(mActions[index].segment(0,2).squaredNorm()));
	// if (ballToLine >= 0.30)
	// {
	// 	reward = exp(-2.0 * (targetP - p).squaredNorm());
	// }
	// else if( (ballP-centerOfGoalpost).norm()-0.20>(p-centerOfGoalpost).norm()  )
	// {
	// 	// cout<<0<<endl;
	// 	reward = exp(-2.0 * (targetP - p).squaredNorm());
	// }
	// else
	// {
	// 	// cout<<1<<endl;
	// 	targetP = centerOfGoalpost + (ballP - centerOfGoalpost) - 0.2 * (ballP - centerOfGoalpost).normalized();
	// 	double curShooting = mActions[index][2];
	// 	if(curShooting >= 1.0)
	// 		curShooting = 1.0;
	// 	double targetShooting = 0.0;

	// 	reward = 1.0 + 8.0 * exp(-2.0 * (targetP - p).squaredNorm()) + 4.0* exp(-2.0 * (ballP - centerOfGoalpost).squaredNorm());// + 4.0 * exp(-(targetShooting-curShooting));
	// 	if(mStates[index][_ID_KICKABLE] == 1 && curShooting>=0)
	// 	{
	// 		Eigen::VectorXd v = mStates[index].segment(_ID_V,2);
	// 		reward += 60.0 * v.dot((targetP - p).normalized());
	// 	}

	// }


	Eigen::VectorXd v = mStates[index].segment(_ID_V, 2);
	Eigen::VectorXd ballV = v + mStates[index].segment(_ID_BALL_V, 2);

	Eigen::VectorXd ballToGoalpost = centerOfGoalpost - ballP;

	reward = 0.1 * exp(-(p-ballP).norm());

	// reward += exp(-ballToGoalpost.norm());

	if(ballV.norm()>0 && ballToGoalpost.norm()>0)
	{
		double cosTheta =  ballV.normalized().dot(ballToGoalpost.normalized());
		reward += ballV.norm()* exp(-acos(cosTheta));
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

	if(index==0)
	{
		if(widthVector.dot(ballPosition)+ballRadius >= mGoalposts[1].second.x())
		{
			if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
			{
				std::cout<<"Red Team GOALL!!"<<std::endl;
				if(!goalRewardPaid[index])
					reward += 180;
				// goalRewardPaid[index] = true;
				mIsTerminalState = true;
			}
		}
		else if(widthVector.dot(ballPosition)-ballRadius < mGoalposts[0].second.x())
		{
			if(abs(heightVector.dot(ballPosition)) < goalpostSize/2.0)
			{
				// std::cout<<"Red Team GOALL!!"<<std::endl;
				// if(!goalRewardPaid[index])
				// 	reward += 10;
				// goalRewardPaid[index] = true;
				mIsTerminalState = true;
			}
		}

	}

	return reward * 0.1;
}



void
Environment::
setHindsightGoal(Eigen::VectorXd randomSchedulerState, Eigen::VectorXd randomSchedulerAction)
{
	int numStates = randomSchedulerState.size()/2;

	mHindsightGoalState = unNormalizeNNState(randomSchedulerState.segment(0, numStates) + randomSchedulerAction.segment(0, numStates));
}

Eigen::VectorXd
Environment::
getHindsightState(Eigen::VectorXd curState)
{
	int numStates = curState.size()/2;
	Eigen::VectorXd combinedState(numStates*2);

	combinedState.segment(0, numStates) = curState.segment(0,numStates);

	combinedState.segment(numStates, numStates) = (normalizeNNState(mHindsightGoalState)- curState.segment(0,numStates)).cwiseProduct(mWGoalStates[0]);
	// hindsightGoal = mWGoalStates;
	return combinedState;
}

double
Environment::
getHindsightReward(Eigen::VectorXd curHindsightState, Eigen::VectorXd schedulerAction)
{
	double goalReward = 0.0;
	int numStates = curHindsightState.size()/2;
	Eigen::VectorXd weightedDistanceToGoal;
	Eigen::VectorXd curState = curHindsightState.segment(0, numStates);
	weightedDistanceToGoal = (mHindsightGoalState-unNormalizeNNState(schedulerAction.segment(0, numStates) + curState)).cwiseProduct(mWGoalStates[0]);
	
	// cout<<mGoalStates[index].segment(_ID_BALL_P, 2).transpose()<<endl;
	// cout<<mSubgoalStates[index].segment(_ID_BALL_P, 2).transpose()<<endl;
	// cout<<mWGoalStates[index].segment(_ID_BALL_P, 2).transpose()<<endl;
	// cout<<weightedDistanceToGoal.segment(_ID_BALL_P,2).transpose()<<endl;
	// cout<<endl;
	if(weightedDistanceToGoal.norm() < 0.40)
	{
		// cout<<weightedDistanceToGoal.norm()<<endl;
		goalReward = 4.0;
		// cout<<"Player "<<0<<" reached Hindsight goal state! "<<weightedDistanceToGoal.norm()<<endl;
		// mIsTerminalState = true;
	}

	return goalReward;
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
			mAccScore[i] += getSchedulerReward(i);
		}

	}

	return rewards;
}

void
Environment::
applyAction(int index)
{
	double maxVel = 4.0;
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
	double maxVel = 4.0;

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

	if(mActions[index].segment(0,2).norm()>1.0)
	{
		mActions[index].segment(0,2) /= mActions[index].segment(0,2).norm();
	}

	// if(index == 0)
	// 	mActions[index][2] = 1.0;

	Eigen::VectorXd applyingForce = mActions[index].segment(0,2);
	// applyingForce /= applyingForce.norm();

	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	
	double reviseStateByTeam = -1;
	if( mCharacters[index]->getTeamName() == mGoalposts[0].first)
		reviseStateByTeam = 1;

	Eigen::VectorXd vel = reviseStateByTeam * skel->getVelocities();

	mForces[index] = 1000.0*reviseStateByTeam*applyingForce;


	// skel->setForces(1000.0*reviseStateByTeam*applyingForce);

	// if (vel.norm() > maxVel)
	// 	vel = vel/vel.norm() * maxVel;
	// skel->setVelocities(reviseStateByTeam * vel);
}


bool
Environment::
isTerminalState()
{
	if(mTimeElapsed>8.0)
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
		mMapStates[i]->reset();
	
		goalRewardPaid[i] = false;
		// if(i == 0 && rand()%2 == 0)
		// {
		// 	ballSkel->setPositions(skelPosition);
		// 	ballSkel->setVelocities(skelVel);
		// }
	}


	mIsTerminalState = false;
	mTimeElapsed = 0;

	mScoreBoard[0] = 0.5;

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

	Eigen::Vector2d centerVector = skel->getPositions();
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

void
Environment::
initGoalState()
{
	mGoalStates.resize(mNumChars);
	mWGoalStates.resize(mNumChars);
	mSubgoalStates.resize(mNumChars);
	mWSubgoalStates.resize(mNumChars);

	bool rewardForGoal = false;
	if(mNumChars == 2)
	{
		for(int i=0;i<mNumChars;i++)
		{
			if(i==0)
			{
				//p.rows() + v.rows() + relativeBallP.rows() + relativeBallV.rows() +
				//ballPossession.rows() + kickable.rows() + otherS.rows() + distanceWall.rows() + goalpostPositions.rows()
				mGoalStates[i] = unNormalizeNNState(getState(i));
				mGoalStates[i].setZero();
				mWGoalStates[i].resize(mGoalStates[i].size());
				mSubgoalStates[i].resize(mGoalStates[i].size());
				mWSubgoalStates[i].resize(mGoalStates[i].size());
				for(int j=0;j<mWGoalStates[i].size();j++)
				{
					mWGoalStates[i][j] = 0.0;
					mSubgoalStates[i][j] = 0.0;
					mWSubgoalStates[i][j] = 0.0;
				}
				Eigen::VectorXd targetBallPosition;

				if(rewardForGoal)
				{
					targetBallPosition = mGoalposts[1].second.segment(0,2);
					Eigen::VectorXd distanceWall(4);
					distanceWall << 4-targetBallPosition[0], 4+targetBallPosition[0], 3-targetBallPosition[1], 3+targetBallPosition[1];

					Eigen::VectorXd goalpostPositions(8);
					goalpostPositions.segment(0,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - targetBallPosition;
					goalpostPositions.segment(2,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - targetBallPosition;
					goalpostPositions.segment(4,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - targetBallPosition;
					goalpostPositions.segment(6,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - targetBallPosition;
				}
				else
				{
					mGoalStates[i].segment(_ID_BALL_P,2) = Eigen::Vector2d(0.0, 0.0);
					mWGoalStates[i].segment(_ID_BALL_P,2) = Eigen::Vector2d(1.0, 1.0);

					// mGoalStates[i][_ID_KICKABLE] = 1.0;
					// mWGoalStates[i][_ID_KICKABLE] = 1.0;
				}


				// Eigen::VectorXd targetBallRelativePosition = Eigen::Vector2d(0.0, 0.0);

			}

			else
			{
				//p.rows() + v.rows() + relativeBallP.rows() + relativeBallV.rows() +
				//ballPossession.rows() + kickable.rows() + otherS.rows() + distanceWall.rows() + goalpostPositions.rows()
				mGoalStates[i] = getState(i);
				mGoalStates[i].setZero();
				mWGoalStates[i].resize(mGoalStates[i].size());
				mSubgoalStates[i].resize(mGoalStates[i].size());
				mWSubgoalStates[i].resize(mGoalStates[i].size());
				for(int j=0;j<mWGoalStates[i].size();j++)
				{
					mWGoalStates[i][j] = 0.0;
					mSubgoalStates[i][j] =0.0;
					mWSubgoalStates[i][j] = 0.0;
				}

				Eigen::VectorXd targetBallPosition;
				if(rewardForGoal)
				{
					targetBallPosition = mGoalposts[1].second.segment(0,2);
					Eigen::VectorXd distanceWall(4);
					distanceWall << 4+targetBallPosition[0], 4-targetBallPosition[0], 3+targetBallPosition[1], 3-targetBallPosition[1];

					Eigen::VectorXd goalpostPositions(8);
					goalpostPositions.segment(0,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - targetBallPosition;
					goalpostPositions.segment(2,2) = mGoalposts[1].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - targetBallPosition;
					goalpostPositions.segment(4,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, 1.5/2.0) - targetBallPosition;
					goalpostPositions.segment(6,2) = mGoalposts[0].second.segment(0,2) + Eigen::Vector2d(0.0, -1.5/2.0) - targetBallPosition;


				}
				else
				{
					mGoalStates[i].segment(_ID_BALL_P,2) = Eigen::Vector2d(0.0, 0.0);
					mWGoalStates[i].segment(_ID_BALL_P,2) = Eigen::Vector2d(1.0, 1.0);

					// mGoalStates[i][_ID_KICKABLE] = 1.0;
					// mWGoalStates[i][_ID_KICKABLE] = 1.0;
				}
				
			}
		}
	}
}

Eigen::VectorXd
Environment::
normalizeNNState(Eigen::VectorXd state)
{
	Eigen::VectorXd normalizedState = state;
	normalizedState.segment(0, 8) = state.segment(0,8)/4.0;
	normalizedState.segment(9, 8) = state.segment(9, 8)/4.0;
	return normalizedState;
}


Eigen::VectorXd
Environment::
unNormalizeNNState(Eigen::VectorXd outSubgoal)
{
	Eigen::VectorXd scaledSubgoal = outSubgoal;
	scaledSubgoal.segment(0, 8) = outSubgoal.segment(0,8)*4.0;
	scaledSubgoal.segment(9, 8) = outSubgoal.segment(9, 8)*4.0;
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
setLinearActorState(int index, Eigen::VectorXd linearActorState)
{
	int numStates = mStates[index].size();
	mSubgoalStates[index] = unNormalizeNNState(linearActorState.segment(0,numStates));
	// mSubgoalStates[index] 
	mWSubgoalStates[index] = softMax(linearActorState.segment(numStates,numStates));


	// for(int i=0;i<mWSubgoalStates[index].size();i++)
	// {
	// 	mWSubgoalStates[index][i] = mWSubgoalStates[index][i];
	// 	if(mWSubgoalStates[index][i] < 0 )
	// 		mWSubgoalStates[index][i] = 0.0;
	// 	if(mWSubgoalStates[index][i] > 1.0 )
	// 		mWSubgoalStates[index][i] = 0.0;
	// }
}


void MapState::setCurState(Eigen::VectorXd curState)
{
	if(isFirst)
	{
		for(int i=0;i<mNumPrev;i++)
		{
			minimaps.push_back(curState);
		}
		isFirst = false;
	}

	if(!updated)
	{
		for(int i=mNumPrev-1;i>0;i--)
		{
			minimaps[i] = minimaps[i-1];
		}
		minimaps[0] = curState;
		updated = true;
	}
	
}
void MapState::endOfStep()
{
	updated = false;
}

void MapState::reset()
{
	minimaps.clear();
	isFirst = true;
	updated = false;
}
std::vector<float> MapState::getVectorizedValue()
{
	std::vector<float> vectorizedValue;
	vectorizedValue.reserve(mNumPrev*minimaps[0].size());
	for(int i=0;i<mNumPrev;i++)
	{
			// std::cout<<minimaps[i].size()<<std::endl;
			// std::cout<<isFirst<<std::endl;
		for(int j=0;j<minimaps[i].size();j++)
		{
			// std::cout<<minimaps[i].size()
			// std::cout<<vectorizedValue.size()<<" "<<i<<" "<<j<<std::endl;

			vectorizedValue.push_back(minimaps[i][j]);
		}
		// cout<<i<<endl;
	}
	// std::cout<<mNumPrev<<std::endl;
	// std::cout<<vectorizedValue.size()<<std::endl;
	// std::cout<<vectorizedValue.capacity()<<std::endl;
	// exit(0);

	return vectorizedValue;
}
