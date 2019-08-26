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
	mWorld->getConstraintSolver()->removeAllConstraints();
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
	mPrevScoreBoard.resize(1);
	mPrevScoreBoard[0] = mScoreBoard[0];
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

	// if(mStates[index][ID_KICKABLE] == 1)
	// {
	// 	// for(int i=0;i<mNumChars;i++)
	// 	// {
	// 	// 	if(ballDistance > (ballSkel->getPositions() - mCharacters[i]->getSkeleton()->getPositions()).norm())
	// 	// 		return;
	// 	// }
	// 	// Eigen::VectorXd relativeVel = skel->getVelocities() - ballSkel->getVelocities();
	// 	ballSkel->setVelocities(skel->getVelocities()*(1.0+me*2.0));
	// 	if(mCharacters[index]->getTeamName() == "A")
	// 		mKicked[0] = 1;
	// 	else
	// 		mKicked[1] = 1;
	// }

	double kickPower = mActions[index][2];

	if(kickPower>=1)
		kickPower = 1;

	if(kickPower >= 0)
	{
		// kickPower = 1.0/(exp(-kickPower)+1);
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

	// std::cout<<mWorld->getConstraintSolver()->getCollisionDetector()-> <<std::endl;
	mTimeElapsed += 1.0 / (double)mSimulationHz;
	// cout<<mTimeElapsed<<endl;

	for(int i=1;i<mCharacters.size();i++)
	{
		for(int j=0;j<i;j++)
		{
			handlePlayerContact(i,j,0.5);
		}
	}

	for(int i=0;i<mCharacters.size();i++)
	{
		// cout<<"aaaaaaaaaaaaa"<<endl;
		handleWallContact(mCharacters[i]->getSkeleton(), 0.08, 0.5);
			// cout<<"bbbbbbbb"<<endl;
	// cout<<"111111111111@"<<endl;
		// cout<<mActions.size()<<endl;
		// handleBallContact(mCharacters[i]->getSkeleton(), 0.08, 1.3);
		if(mStates[i][ID_KICKABLE] == 1)
			handleBallContact(i, 0.12, 2.0);
				// cout<<"cccccccccccc"<<endl;
	// cout<<"222222222222@"<<endl;
		// handleContact(mCharacters[i]->getSkeleton(), 0.08, 0.5);
	}
	// handleContact(ballSkel, 0.08);
	
	// cout<<"@@@@@@@@@@@@@"<<endl;
	handleWallContact(ballSkel, 0.08, 0.8);
	// cout<<"1111"<<endl;
	boundBallVelocitiy(15.0);

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

	Eigen::VectorXd distanceWall(4);


	// if(index == 0)
	// {
	// 	cout<<distanceWall.transpose()<<endl;
	// }

	// Ball's state
	Eigen::Vector2d ballP, ballV;
	ballP = ballSkel->getPositions() - p;
	ballV = ballSkel->getVelocities() - v;

	distanceWall << 4-ballP[0], -4-ballP[0], 3-ballP[1], -3-ballP[1];

	Eigen::VectorXd ballPossession(1);
	if(index == 0 || index == 1)
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

	bool useMap = false;
	std::vector<double> mapState;
	if(useMap)
	{
		mapState = getStateMinimap(index);
	}

	Eigen::VectorXd vecState(p.rows() + v.rows() + ballP.rows() + ballV.rows() +
	ballPossession.rows() + kickable.rows());


	vecState<<p,v,ballP,ballV,ballPossession,kickable;
	mStates[index] = vecState;
	// time_check_end();
	// if(index == 1)
	// 	cout<<vecState.segment(ID_BALL_P, 2).transpose()<<endl;


	std::vector<double> state;
	if(useMap)
	{
		state.resize(ballPossession.rows() + kickable.rows() + mapState.size());
	}
	else
	{
		state.resize(p.rows() + v.rows() + ballP.rows() + ballV.rows() +
			ballPossession.rows() + kickable.rows() + otherS.rows() + distanceWall.rows());
	}

	
	count = 0;
	if(!useMap)
	{
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
	}
	for(int i=0;i<ballPossession.rows();i++)
	{
		state[count++] = ballPossession[i];
	}
	for(int i=0;i<kickable.rows();i++)
	{
		state[count++] = kickable[i];
	}

	if(useMap)
	{
		for(int i=0;i<mapState.size();i++)
		{
			state[count++] = mapState[i];
		}
	}
	else
	{
		for(int i=0;i<otherS.rows();i++)
		{
			state[count++] = otherS[i];
		}
		for(int i=0;i<distanceWall.rows();i++)
		{
			state[count++] = distanceWall[i];
		}

	}
	

	// for(int i=0;i<10;i++)
	// {
	// 	if(std::isnan(state[i]))
	// 	{
	// 		cout<<"get State : nan occured! "<<i<<endl;
	// 	}
	// }

	// 
	// if(index == 0)
	// {
	// 	std::cout<<mStates[index].segment(ID_BALL_P,2).norm()<<endl;
	// }


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

std::vector<double>
Environment::
getStateMinimap(int index)
{
	// std::vector<Eigen::MatrixXd> minimaps;

	std::vector<double> minimaps;
	minimaps.resize(40*40*5);
	int count =0;
	int numRows = 40;
	int numCols = 40; 

	/// map for wall
	{
		// Eigen::MatrixXd minimap(numRows, numCols);
		// minimap.setZero();
		std::vector<double> minimap;
		minimap.resize(numRows*numCols);

		for(int i=0;i<numCols;i++)
		{
			for(int j=0;j<numRows;j++)
			{
				minimap[i*numRows+j] = 0.5;
			}	
		}
		for(double i = -4;i<4;i+= 0.1)
		{
			for(double j=-3;j<3;j+=0.1)
			{
				std::pair<int, int> pixel = getPixelFromPosition(i, j);
				minimap[pixel.second*numRows+pixel.first] = 0.0;
			}
		}

		for(int i=0;i<minimap.size();i++)
		{
			minimaps[count++] = minimap[i];
		}

		// minimaps.push_back(minimap);

	}
	// try
	// {
	//     *(int*) 0 = 0;
	// }
	// catch (std::exception& e)
	// {
	//     std::cerr << "Exception caught : " << e.what() << std::endl;
	// }
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

	/// map for ball
	{
		std::vector<double> minimap;
		minimap.resize(numRows*numCols);

		Eigen::VectorXd position = ballSkel->getPositions();

		std::pair<int, int> pixel = getPixelFromPosition(position[0], position[1]);
		minimap[pixel.second*numRows+pixel.first] = 1.0;
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
	// 	if(std::isnan(minimaps[i]))
	// 	{
	// 		cout<<"Nan is occured at "<<i/(numRows*numCols)<<endl;
	// 	}
	// }
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

	// visualizeMinimap(minimaps);
	// exit(0);


	return minimaps;
}

std::vector<double>
Environment::
getStateMinimapRGB(int index)
{
	// std::vector<Eigen::MatrixXd> minimaps;

	std::vector<double> minimaps;
	minimaps.resize(40*40*1);
	int count =0;
	int numRows = 40;
	int numCols = 40; 

	/// map for wall
	{
		// Eigen::MatrixXd minimap(numRows, numCols);
		// minimap.setZero();
		std::vector<double> minimap;
		minimap.resize(numRows*numCols);

		for(int i=0;i<numCols;i++)
		{
			for(int j=0;j<numRows;j++)
			{
				minimap[i*numRows+j] = 0.5;
			}	
			
		}
		for(double i = -4;i<4;i+= 0.1)
		{
			for(double j=-3;j<3;j+=0.1)
			{
				std::pair<int, int> pixel = getPixelFromPosition(i, j);
				minimap[pixel.second*numRows+pixel.first] = 0.0;
			}
		}
		Eigen::VectorXd position = mCharacters[index]->getSkeleton()->getPositions();

		std::pair<int, int> pixel = getPixelFromPosition(position[0], position[1]);
		minimap[pixel.second*numRows+pixel.first] = 1.0;

		for(int i=0;i<minimap.size();i++)
		{
			minimaps[count++] = minimap[i];
		}

		// minimaps.push_back(minimap);

	}
	// try
	// {
	//     *(int*) 0 = 0;
	// }
	// catch (std::exception& e)
	// {
	//     std::cerr << "Exception caught : " << e.what() << std::endl;
	// }
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

	/// map for ball
	{
		std::vector<double> minimap;
		minimap.resize(numRows*numCols);

		Eigen::VectorXd position = ballSkel->getPositions();

		std::pair<int, int> pixel = getPixelFromPosition(position[0], position[1]);
		minimap[pixel.second*numRows+pixel.first] = 1.0;
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
	// 	if(std::isnan(minimaps[i]))
	// 	{
	// 		cout<<"Nan is occured at "<<i/(numRows*numCols)<<endl;
	// 	}
	// }
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

	// visualizeMinimap(minimaps);
	// exit(0);


	return minimaps;
}

Eigen::VectorXd
Environment::
updateScoreBoard(std::string teamName)
{
	mPrevScoreBoard = mScoreBoard;
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

	double rewardScale = 0.2;


	if(mCharacters[index]->getTeamName()=="A")
	{
		reward += (mNumIterations/50.0) * rewardScale * (2*mScoreBoard[0] -1);
		// reward += (mNumIterations/50.0) * 1.0 * (mScoreBoard[0]);
	}
	else
	{
		reward += (mNumIterations/50.0) * rewardScale * (1-2*mScoreBoard[0]);
		// reward += (mNumIterations/50.0) * 1.0 * (1-mScoreBoard[0]);
	}

	
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

	// reward += 1.0 * exp(-pow(myDistanceBall,2.0));

	reward += rewardScale * (1 - mNumIterations/50.0) * exp(-pow(myDistanceBall,2.0));

	// reward += -0.5 * (mNumIterations/50.0) * exp(-pow(oppMinDistance,2.0));

	if(mPrevScoreBoard[0] == mScoreBoard[0] + 1)
	{
		if(mCharacters[index]->getTeamName()=="A")
		{
			reward += -30 * (mNumIterations/50.0) * rewardScale;
		}
		else 
		{
			reward += 30 * (mNumIterations/50.0) * rewardScale;
		}
	}
	else if(mPrevScoreBoard[0] == mScoreBoard[0] - 1)
	{
		if(mCharacters[index]->getTeamName()=="A")
		{
			reward += 30 * (mNumIterations/50.0) * rewardScale;
		}
		else 
		{
			reward += -30 * (mNumIterations/50.0) * rewardScale;
		}
	}



	// reward += 0.1 * (1 - mNumIterations/400.0) * exp(-pow(myDistanceBall,2.0));

	// reward += rewardScale * exp(-pow(myDistanceBall,2.0));
	// if(index == 0)
	// {
	// 	cout<<reward<<endl;
	// }
	// reward += 0.1 * exp(-pow(myDistanceBall,2.0));

	// if(abs(oppMinDistance - myTeamMinDistance)>0.01)

	// reward += (mNumIterations/50.0) * rewardScale/2.0 * (oppMinDistance - myTeamMinDistance);




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

	for(int i=0;i<2;i++)
	{
		if(mActions[index][i] > 1.0)
			mActions[index][i] = 1.0;
		if(mActions[index][i] < -1.0)
			mActions[index][i] = -1.0;
	}



	double maxVel = 4.0;
	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	Eigen::VectorXd vel = skel->getVelocities();

	vel += 0.5*mActions[index].segment(0, vel.size());
	// cout<<vel.transpose()<<endl;

	// for(int i=0;i<vel.size();i++)
	// {
	// 	if(abs(vel[i])>maxVel)
	// 	{
	// 		vel[i] /= abs(vel[i]);
	// 		vel[i] *= maxVel;
	// 	}
	// }

	// for(int i=0;i<mActions[index].size();i++)
	// {
	// 	if(std::isnan(mActions[index][i]))
	// 	{
	// 		cout<<"NAN OCCURED IN SET ACTION ! "<<endl;
	// 	}
	// }

	if (vel.norm() > maxVel)
		vel = vel/vel.norm() * maxVel;

	if (mStates[index][ID_POSSESSION] == 1)
	{
		// mActions[index][2] = -1.0;

		// vel = vel*0.5;

		if (vel.norm() > 3.0)
			vel = vel/vel.norm() * 3.0;

		// vel.setZero();
	}
	// mActions[index][2] = 1.0;
	

	// if(index == 1)
	// {
	// 	cout<<a.segment(0, vel.size()).transpose()<<endl;
	// }
	skel->setVelocities(vel);

}


bool
Environment::
isTerminalState()
{
	if(mTimeElapsed>15.0)
	{
		cout<<"Time overed"<<endl;
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