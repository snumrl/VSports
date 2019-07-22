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
	mAccScore.resize(mNumChars);
	mAccScore.setZero();
	mStates.resize(mNumChars);


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

	for(int i=0;i<mNumChars;i++)
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

	if(mStates[index][ID_KICKABLE] == 1)
	{
		// for(int i=0;i<mNumChars;i++)
		// {
		// 	if(ballDistance > (ballSkel->getPositions() - mCharacters[i]->getSkeleton()->getPositions()).norm())
		// 		return;
		// }
		// Eigen::VectorXd relativeVel = skel->getVelocities() - ballSkel->getVelocities();
		ballSkel->setVelocities(skel->getVelocities()*(1.0+me*2.0));
		if(mCharacters[index]->getTeamName() == "A")
			mKicked[0] = 1;
		else
			mKicked[1] = 1;
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
step()
{
	// std::cout<<ballSkel->getCOM().transpose()<<std::endl;
	// std::cout<<mCharacters[0]->getSkeleton()->getVelocities().transpose()<<std::endl;
		// cout<<"!!!!!!!!!!!!!!!!!!!"<<endl;
	for(int i=0;i<mCharacters.size();i++)
	{
		// cout<<"aaaaaaaaaaaaa"<<endl;
		handleWallContact(mCharacters[i]->getSkeleton(), 0.08, 0.5);
			// cout<<"bbbbbbbb"<<endl;
	// cout<<"111111111111@"<<endl;
		// cout<<mActions.size()<<endl;
		// handleBallContact(mCharacters[i]->getSkeleton(), 0.08, 1.3);
		if(mActions[i][2] > 0.0)
			handleBallContact(i, 0.08, 1.3);
				// cout<<"cccccccccccc"<<endl;
	// cout<<"222222222222@"<<endl;
		// handleContact(mCharacters[i]->getSkeleton(), 0.08, 0.5);
	}
	// handleContact(ballSkel, 0.08);
	
	// cout<<"@@@@@@@@@@@@@"<<endl;
	handleWallContact(ballSkel, 0.08, 0.8);
	// cout<<"1111"<<endl;
	boundBallVelocitiy(9.0);

	// cout<<"22222"<<endl;

	// ballSkel->setVelocities(ballSkel->getVelocities()*0.90);
	Eigen::VectorXd ballForce = ballSkel->getForces();
	// cout<<ballForce.transpose()<<endl;
	// cout<<ballSkel->getVelocities().transpose()<<endl;
	// cout<<endl;
	ballForce -= 2*ballSkel->getVelocities();
	ballSkel->setForces(ballForce);
	// cout<<"3333"<<endl;
	// for(int i=0;i<mCharacters.size();i++)
	// {
	// 	handleContact(mCharacters[i]->getSkeleton(), 0.08, 0.5);
	// }
	// handleContact(ballSkel, 0.08);
	
	// cout<<"44444"<<endl;

	// check the reward for debug
	updateScoreBoard();

	getRewards();

	mWorld->step();
	// cout<<"55555555"<<endl;	
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
/*
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

	// Eigen::VectorXd distanceWall(4);

	// distanceWall << 4-p[0], -4-p[0], 3-p[1], -3-p[1];

	// if(index == 0)
	// {
	// 	cout<<distanceWall.transpose()<<endl;
	// }

	// Ball's state
	Eigen::Vector2d ballP, ballV;
	ballP = ballSkel->getPositions() - p;
	ballV = ballSkel->getVelocities() - v;

	// Observation
	std::string teamName = character->getName().substr(0,1);
	
	// Do not count for cur Character
	// double distance[mCharacters.size()-1];
	// int distanceIndex[mCharacters.size()-1];
	// int count =0;

	// // We will get the other's position & velocities in sorted form.
	// for(int i=0;i<mCharacters.size();i++)
	// {
	// 	if(i != index)
	// 	{
	// 		distanceIndex[count] = i;
	// 		Eigen::Vector2d curP = mCharacters[i]->getSkeleton()->getPositions();
	// 		distance[count] = (curP-p).norm();
	// 		count++;
	// 	}
	// }
	// count = 0;

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
	if(ballP.norm()<0.15+0.08)
	{
		kickable[0] = 1;
	}
	else
	{
		kickable[0] = 0;
	}

	// otherS.resize(0);
	
	// Fill in the goal basket's relational position
	// Eigen::VectorXd goalpostPositions(4);
	// if(teamName == mGoalposts[0].first)
	// {
	// 	goalpostPositions.segment(0,2) = mGoalposts[0].second.segment(0,2) - p;
	// 	goalpostPositions.segment(2,2) = mGoalposts[1].second.segment(0,2) - p;
	// }
	// else
	// {
	// 	goalpostPositions.segment(0,2) = mGoalposts[1].second.segment(0,2) - p;
	// 	goalpostPositions.segment(2,2) = mGoalposts[0].second.segment(0,2) - p;
	// }



	time_check_start();

	// Put these arguments in a single vector

	Eigen::VectorXd mapState = getStateMinimap(index);

	Eigen::VectorXd s(p.rows() + v.rows() + ballP.rows() + ballV.rows() +
	ballPossession.rows() + kickable.rows() + mapState.rows());


	// Eigen::VectorXd s(p.rows() + v.rows() + ballP.rows() + ballV.rows() + 1);

	// s.segment(0, p.rows()) = p;
	// s.segment(p.rows(), v.rows()) = v;
	// s.segment(p.rows() + v.rows(), 1)[0] = isDribbler;
	// s.segment(p.rows() + v.rows() + 1, otherS.size()) = otherS;
	// s.segment(p.rows() + v.rows() + 1 + otherS.size(), 4) = goalpostPositions;
	s<<p,v,ballP,ballV,ballPossession,kickable, mapState;
	mStates[index] = s;
	time_check_end();
	return s;
}
*/
std::vector<double>
Environment::
getState(int index)
{
	// Character's state
	Eigen::Vector2d p,v;
	Character2D* character = mCharacters[index];
	p = character->getSkeleton()->getPositions();
	v = character->getSkeleton()->getVelocities();
	// int isDribbler = (getDribblerIndex()==index) ? 1 : 0;

	// Eigen::VectorXd distanceWall(4);

	// distanceWall << 4-p[0], -4-p[0], 3-p[1], -3-p[1];

	// if(index == 0)
	// {
	// 	cout<<distanceWall.transpose()<<endl;
	// }

	// Ball's state
	Eigen::Vector2d ballP, ballV;
	ballP = ballSkel->getPositions() - p;
	ballV = ballSkel->getVelocities() - v;

	// Observation
	std::string teamName = character->getName().substr(0,1);
	
	// Do not count for cur Character
	// double distance[mCharacters.size()-1];
	// int distanceIndex[mCharacters.size()-1];
	// int count =0;

	// // We will get the other's position & velocities in sorted form.
	// for(int i=0;i<mCharacters.size();i++)
	// {
	// 	if(i != index)
	// 	{
	// 		distanceIndex[count] = i;
	// 		Eigen::Vector2d curP = mCharacters[i]->getSkeleton()->getPositions();
	// 		distance[count] = (curP-p).norm();
	// 		count++;
	// 	}
	// }
	// count = 0;

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
	if(ballP.norm()<0.15+0.08)
	{
		kickable[0] = 1;
	}
	else
	{
		kickable[0] = 0;
	}

	
	// Fill in the goal basket's relational position
	// Eigen::VectorXd goalpostPositions(4);
	// if(teamName == mGoalposts[0].first)
	// {
	// 	goalpostPositions.segment(0,2) = mGoalposts[0].second.segment(0,2) - p;
	// 	goalpostPositions.segment(2,2) = mGoalposts[1].second.segment(0,2) - p;
	// }
	// else
	// {
	// 	goalpostPositions.segment(0,2) = mGoalposts[1].second.segment(0,2) - p;
	// 	goalpostPositions.segment(2,2) = mGoalposts[0].second.segment(0,2) - p;
	// }



	// time_check_start();

	// Put these arguments in a single vector

	std::vector<double> mapState = getStateMinimap(index);

	Eigen::VectorXd vecState(p.rows() + v.rows() + ballP.rows() + ballV.rows() +
	ballPossession.rows() + kickable.rows());


	vecState<<p,v,ballP,ballV,ballPossession,kickable;
	mStates[index] = vecState;
	// time_check_end();


	std::vector<double> state;
	state.resize(p.rows() + v.rows() + ballP.rows() + ballV.rows() +
	ballPossession.rows() + kickable.rows() + mapState.size());
	int count = 0;
	for(int i=0;i<p.rows();i++)
	{
		state[count++] = p[i];
	}
	for(int i=0;i<v.rows();i++)
	{
		state[count++] = v[i];
	}
	for(int i=0;i<ballP.rows();i++)
	{
		state[count++] = ballP[i];
	}
	for(int i=0;i<ballV.rows();i++)
	{
		state[count++] = ballV[i];
	}
	for(int i=0;i<ballPossession.rows();i++)
	{
		state[count++] = ballPossession[i];
	}
	for(int i=0;i<kickable.rows();i++)
	{
		state[count++] = kickable[i];
	}
	for(int i=0;i<mapState.size();i++)
	{
		state[count++] = mapState[i];
	}
	// cout<<count<<endl;
	// exit(0);

	// time_check_end();
	return state;
}
// Makes 4 layers
/*
	---------------------
	|					|
	|					|
	|					|
	---------------------


*/

std::pair<int, int> getPixelFromPosition(double x, double y, double maxX = 4.0, double maxY = 3.0, 
													int numRows = 40, int numCols = 40)
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

std::vector<double>
Environment::
getStateMinimap(int index)
{
	// std::vector<Eigen::MatrixXd> minimaps;

	std::vector<double> minimaps;
	minimaps.resize(40*40*4);
	int count =0;
	int numRows = 40;
	int numCols = 40; 

	/// map for wall
	{
		// Eigen::MatrixXd minimap(numRows, numCols);
		// minimap.setZero();
		std::vector<double> minimap;
		minimap.resize(numRows*numCols);

		for(double i = -4;i<4;i+= 0.1)
		{
			double j;
			j = -3;
			std::pair<int, int> pixel = getPixelFromPosition(i, j);
			minimap[pixel.second*numRows+pixel.first] = 1.0;

			j = 3;
			pixel = getPixelFromPosition(i, j);
			minimap[pixel.second*numRows+pixel.first] = 1.0;
		}

		for(double j = -3;j<3;j+= 0.1)
		{
			double i;
			i = -4;
			std::pair<int, int> pixel = getPixelFromPosition(i, j);
			minimap[pixel.second*numRows+pixel.first] = 1.0;

			i = 4;
			pixel = getPixelFromPosition(i, j);
			minimap[pixel.second*numRows+pixel.first] = 1.0;
		}

		// for(int i=0;i<numCols;i++)
		// {
		// 	for(int j=0;j<numRows;j++)
		// 	{
		// 		minimaps[count] = minimap(i,j);
		// 		count++;
		// 	}
		// }
		for(int i=0;i<minimap.size();i++)
		{
			minimaps[count++] = minimap[i];
		}

		// minimaps.push_back(minimap);

	}

	/// map for me
	{
		std::vector<double> minimap;
		minimap.resize(numRows*numCols);

		Eigen::VectorXd position = mCharacters[index]->getSkeleton()->getPositions();

		std::pair<int, int> pixel = getPixelFromPosition(position[0], position[1]);
		minimap[pixel.second*numRows+pixel.first] = 1.0;
		// std::string teamName = mCharacters[index]->getTeamName();
		for(int i=0;i<minimap.size();i++)
		{
			minimaps[count++] = minimap[i];
		}
	}

	/// map for My Team
	{
		std::vector<double> minimap;
		minimap.resize(numRows*numCols);

		std::string mTeamName = mCharacters[index]->getTeamName();
		for(int i=0;i<mCharacters.size();i++)
		{
			if(i != index)
			{
				if(mCharacters[i]->getTeamName()==mTeamName)
				{
					Eigen::VectorXd position = mCharacters[i]->getSkeleton()->getPositions();

					std::pair<int, int> pixel = getPixelFromPosition(position[0], position[1]);
					minimap[pixel.second*numRows+pixel.first] = 1.0;
				}
			}

		}


		// std::string teamName = mCharacters[index]->getTeamName();
		for(int i=0;i<minimap.size();i++)
		{
			minimaps[count++] = minimap[i];
		}
	}

	/// map for Opponent Team
	{
		std::vector<double> minimap;
		minimap.resize(numRows*numCols);

		std::string mTeamName = mCharacters[index]->getTeamName();
		for(int i=0;i<mCharacters.size();i++)
		{
			if(mCharacters[i]->getTeamName()!=mTeamName)
			{
				Eigen::VectorXd position = mCharacters[i]->getSkeleton()->getPositions();

				std::pair<int, int> pixel = getPixelFromPosition(position[0], position[1]);
				minimap[pixel.second*numRows+pixel.first] = 1.0;
			}

		}
		for(int i=0;i<minimap.size();i++)
		{
			minimaps[count++] = minimap[i];
		}
	}


	// for(int i=0;i<minimaps.size();i++)
	// {
	// 	for(int col=0;col<numCols;col++)
	// 	{
	// 		for(int row=0;row<numCols;row++)
	// 		{
	// 			cout<<minimaps[i](col,row)<<" ";
	// 		}
	// 		cout<<endl;
	// 	}
	// 	cout<<"#################################################"<<endl;
	// }


	return minimaps;
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

	if(mCharacters[index]->getTeamName()=="A")
	{
		reward += (mNumIterations/400.0) * 0.1 * (2*mScoreBoard[0] -1);
	}
	else
	{
		reward += (mNumIterations/400.0) * 0.1 * (1-2*mScoreBoard[0]);
	}


	// reward += 1.0 * mKicked[index];
	
	double myTeamMinDistance = FLT_MAX;
	double oppMinDistance = FLT_MAX;

	for(int i=0;i<mCharacters.size();i++)
	{
		if(mCharacters[i]->getTeamName() == mCharacters[index]->getTeamName())
		{
			double distanceBall = mStates[i].segment(ID_BALL_P, 2).norm();
			if(myTeamMinDistance > distanceBall)
			{
				myTeamMinDistance = distanceBall;
			}
		}
		else
		{
			double distanceBall = mStates[i].segment(ID_BALL_P, 2).norm();
			if(oppMinDistance > distanceBall)
			{
				oppMinDistance = distanceBall;
			}
		}
	}
	double myDistanceBall = mStates[index].segment(ID_BALL_P, 2).norm();

	reward += (1-mNumIterations/400.0) * 0.1 * exp(-pow(myDistanceBall,2.0));

	if(abs(oppMinDistance - myTeamMinDistance)>0.01)
		reward += (mNumIterations/400.0) * 0.05 * (oppMinDistance - myTeamMinDistance)/abs(oppMinDistance - myTeamMinDistance);




	// reward += -(1-mNumIterations/300.0) * 0.1 * exp(-pow((mCharacters[1-index]->getSkeleton()->getPositions() - ballPosition).norm(),2.0));



	// cout<<mNumIterations<<endl;
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

// used only in test window
std::vector<double> 
Environment::
getRewards()
{
	// mNumIterations = 400.0;
	std::vector<double> rewards;
	// for(int i=0;i<mCharacters.size();i++)
	for(int i=0;i<mNumChars;i++)
	{
		// cout<<i<<endl;
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
	mActions[index] = a;
	mTimeElapsed += 1.0 / (double)mControlHz;
	double maxVel = 3.0;
	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	Eigen::VectorXd vel = skel->getVelocities();

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

}


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