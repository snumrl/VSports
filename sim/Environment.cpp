#include "Environment.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include "./BehaviorTree.h"
#include "dart/external/lodepng/lodepng.h"
#include "AgentEnvWindow.h"
#include "../motion/BVHmanager.h"
#include "../utils/Utils.h"
#include "../extern/ICA/plugin/MotionGenerator.h"
#include <iostream>
#include <chrono>
#include <random>
#include <ctime>
#include <signal.h>
#include <dart/utils/utils.hpp>

using namespace std;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::collision;
using namespace dart::constraint;

// p.rows() + v.rows() + ballP.rows() + ballV.rows() +
// 	ballPossession.rows() + kickable.rows() + goalpostPositions.rows()

Environment::
Environment(int control_Hz, int simulation_Hz, int numChars, std::string bvh_path, std::string nn_path)
:mControlHz(control_Hz), mSimulationHz(simulation_Hz), mNumChars(numChars), mWorld(std::make_shared<dart::simulation::World>()),
mIsTerminalState(false), mTimeElapsed(0), mNumIterations(0), mSlowDuration(180), mNumBallTouch(0), endTime(15), prevContact(false), curContact(false),
criticalPointFrame(0), curFrame(0)
{
	srand((unsigned int)time(0));
	initBall();
	initGoalposts();
	initFloor();
	// getNumState();

	mWorld->setTimeStep(1.0/mSimulationHz);

	// cout<<mWorld->getTimeStep()<<endl;
	// cout<<"11111"<<endl;
	// mWorld->getConstraintSolver()->removeAllConstraints();
	// cout<<"2222"<<endl;
	// mWindow = new AgentEnvWindow(0, this);
	// cout<<"3333"<<endl;
	prevBallPositions.resize(3);
	for(int i=0;i<prevBallPositions.size();i++)
	{
		prevBallPositions[i].setZero();
	}
	mTargetBallPosition.setZero();
	this->endTime = 1000;

	this->initCharacters(bvh_path);
	this->initMotionGenerator(nn_path);
    mWorld->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));
	this->reset();
	this->criticalPoint_targetBallPosition = Eigen::Vector3d(0.0, 0.85, 0.0);
	this->criticalPoint_targetBallVelocity = Eigen::Vector3d(0.0, 0.0, 0.0);
}

void
Environment::
initMotionGenerator(std::string dataPath)
{
	initDartNameIdMapping();
	mMotionGenerator = new ICA::dart::MotionGenerator(dataPath, this->dartNameIdMap);

	Eigen::VectorXd targetZeroVec(19);
	targetZeroVec.setZero();

	BVHmanager::setPositionFromBVH(mCharacters[0]->getSkeleton(), mBvhParser, 50);
	Eigen::VectorXd bvhPosition = mCharacters[0]->getSkeleton()->getPositions();
	mMotionGenerator->setCurrentPose(bvhPosition, Utils::toStdVec(targetZeroVec));
	mCharacters[0]->getSkeleton()->setPositions(bvhPosition);
	curFrame++;
}

void
Environment::
initDartNameIdMapping()
{    
	SkeletonPtr bvhSkel = mCharacters[0]->getSkeleton();
	int curIndex = 0;
	for(int i=0;i<bvhSkel->getNumBodyNodes();i++)
	{
		this->dartNameIdMap[bvhSkel->getBodyNode(i)->getName()] = curIndex;
		curIndex += bvhSkel->getBodyNode(i)->getParentJoint()->getNumDofs();
	}
}



	// Create A team, B team players.
void
Environment::
initCharacters(std::string bvhPath)
{
	for(int i=0;i<mNumChars;i++)
	{
		mCharacters.push_back(new Character3D("A_" + to_string(i)));
	}
	mBvhParser = new BVHparser(bvhPath.data(), BVHType::BASKET);
	mBvhParser->writeSkelFile();

	SkeletonPtr bvhSkel = dart::utils::SkelParser::readSkeleton(mBvhParser->skelFilePath);
	// cout<<charNames[0]<<endl;


	// in the case of single player
	mCharacters[0]->mSkeleton = bvhSkel;


	BVHmanager::setPositionFromBVH(bvhSkel, mBvhParser, 0);
	mWorld->addSkeleton(bvhSkel);



	mActions.resize(mNumChars);
	mStates.resize(mNumChars);
	mLocalStates.resize(mNumChars);

	// initBehaviorTree();
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
	floorSkel = SkelHelper::makeBasketBallFloor();
	// floorSkel = SkelHelper::makeFloor();
	mWorld->addSkeleton(floorSkel);
	setSkelCollidable(floorSkel, true);
}

void 
Environment::
initBall()
{
	ballSkel = SkelHelper::makeBall();
	setSkelCollidable(ballSkel, true);
	mWorld->addSkeleton(ballSkel);
}

void
Environment::
handlePlayerContact(int index1, int index2, double me)
{

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

	// No simulation for now
	// mWorld->step();
}


void
Environment::
stepAtOnce()
{
	// cout<<"Start"<<endl;

	for(int i=0;i<mCharacters.size();i++)
	{
		applyAction(i);
	}
	


	int sim_per_control = this->getSimulationHz()/this->getControlHz();
	for(int i=0;i<sim_per_control;i++)
	{
		this->step();
	}
	curFrame++;

	if(isTerminalState())
	{
		sleep(2000);
		reset();
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
	Eigen::VectorXd state;
	// state.setZero();

	// Use the same format of the motion learning
	std::vector<double> _ICAPosition;
	Motion::MotionSegment* ms = mMotionGenerator->motionGenerators[0]->mMotionSegment;
    MotionRepresentation::getData(ms, _ICAPosition, ms->mPoses.size()-1);
	Eigen::VectorXd ICAPosition = Utils::toEigenVec(_ICAPosition);

	Motion::Root root = ms->getLastPose()->getRoot();

	Eigen::Isometry3d baseToRoot = mMotionGenerator->getBaseToRootMatrix(root);
	Eigen::Vector3d relCurBallPosition = baseToRoot.inverse()*curBallPosition;

	ICAPosition.segment(MotionRepresentation::posOffset,3) = relCurBallPosition;


	// Get goalpost position
	Eigen::Vector3d relTargetPosition;

	relTargetPosition = baseToRoot.inverse()*(mTargetBallPosition*100.0);
	// std::cout<<"##############"<<std::endl;
	// std::cout<<mTargetBallPosition<<std::endl;
	// std::cout<<relTargetPosition<<std::endl;

	state.resize(ICAPosition.rows() + relTargetPosition.rows());
	
	int curIndex = 0;
	for(int i=0;i<ICAPosition.rows();i++)
	{
		state[curIndex] = ICAPosition[i];
		curIndex++;
	}
	for(int i=0;i<relTargetPosition.rows();i++)
	{
		state[curIndex] = relTargetPosition[i];
		curIndex++;
	}



	mStates[index] = state;
	// cout<<"getState end"<<endl;
	return state;
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
	return getState(index);
}

double
Environment::
getReward(int index, bool verbose)
{
	double reward = 0;

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

	auto nextPositionAndContacts = mMotionGenerator->generateNextPoseAndContacts(Utils::toStdVec(mActions[index]));
    Eigen::VectorXd nextPosition = nextPositionAndContacts.first;
    Eigen::Vector4d nextContacts = nextPositionAndContacts.second.segment(0,4);





    int curActionType = 0;
    for(int i=4;i<12;i++)
    {
        if(mActions[index][i] >= 0.5)
            curActionType = i-4;
    }

    //Update ball Positions
   	updatePrevBallPositions(nextPositionAndContacts.second.segment(4,3));

    //Update hand Contacts;
    updatePrevContacts(nextPositionAndContacts.second.segment(2,2));
    




    // if(curActionType == 1 || curActionType == 3)
    // {
    	//Let's check if this is critical point or not
    	if(prevContact && !curContact)
    	{
            this-> criticalPoint_targetBallPosition = this->prevBallPositions[0];
            this-> criticalPoint_targetBallVelocity = (this->prevBallPositions[0] - this->prevBallPositions[2])*15*1.5;
            // std::cout<<"this->prevBallPosition : "<<this->prevBallPosition.transpose()<<std::endl;
            // std::cout<<"this->pprevBallPosition : "<<this->pprevBallPosition.transpose()<<std::endl;
            // std::cout<<"this->ppprevBallPosition : "<<this->ppprevBallPosition.transpose()<<std::endl;
            
            criticalPointFrame = curFrame;
    	}

    // }


	if(curActionType != 0 && !curContact)
	{
		curBallPosition = computeBallPosition();
	}


    Eigen::Vector6d ballPosition;
    ballPosition.setZero();

    ballPosition.segment(3,3) = curBallPosition;



    mCharacters[0]->mSkeleton->setPositions(nextPosition);
    mCharacters[0]->mSkeleton->setVelocities(mCharacters[0]->mSkeleton->getVelocities().setZero());



    ballSkel->setPositions(ballPosition);

    Eigen::Vector6d zeroVelocity;
    zeroVelocity.setZero();
    ballSkel->setVelocities(zeroVelocity);

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

	if((mCharacters[0]->getSkeleton()->getCOM()-mTargetBallPosition).norm() > 100.0)
		mIsTerminalState = true;

	return mIsTerminalState;
}

void
Environment::
reset()
{

	mIsTerminalState = false;
	mTimeElapsed = 0;

	mAccScore.setZero();
	mNumBallTouch = 0;
	resetTargetBallPosition();
	resetCharacterPositions();
}


void
Environment::
resetTargetBallPosition()
{
	double xRange = 28.0*0.8*0.5*0.8;
	double zRange = 15.0*0.8*0.5*0.8;

	mTargetBallPosition[0] = (double) rand()/RAND_MAX * xRange*2.0 - xRange;
	mTargetBallPosition[1] = 0.0;
	mTargetBallPosition[2] = (double) rand()/RAND_MAX * zRange*2.0 - zRange;
}

void 
Environment::
resetCharacterPositions()
{
	double xRange = 28.0*0.8*0.5*0.8;
	double zRange = 15.0*0.8*0.5*0.8;	



	Eigen::VectorXd standPosition = mCharacters[0]->getSkeleton()->getPositions();
	standPosition[4] = 0.895;

	standPosition[3] = (double) rand()/RAND_MAX * xRange*2.0 - xRange;
	standPosition[5] = (double) rand()/RAND_MAX * zRange*2.0 - zRange;

	Eigen::VectorXd targetZeroVec(19);
	targetZeroVec.setZero();

	mMotionGenerator->setCurrentPose(standPosition, Utils::toStdVec(targetZeroVec));


	for(int i=0;i<mNumChars;i++)
	{
		mCharacters[i]->getSkeleton()->setPositions(standPosition);
	}

}
Eigen::VectorXd
Environment::
normalizeNNState(Eigen::VectorXd state)
{
	Eigen::VectorXd normalizedState = state;

	return normalizedState;
}


Eigen::VectorXd
Environment::
unNormalizeNNState(Eigen::VectorXd normalizedState)
{
	Eigen::VectorXd unNormalizedState = normalizedState;
	return unNormalizedState;
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
updatePrevBallPositions(Eigen::Vector3d newBallPosition)
{
	for(int i=2;i>=1;i--)
    {
    	prevBallPositions[i] =prevBallPositions[i-1];
    }
    prevBallPositions[0] = curBallPosition;
    curBallPosition = newBallPosition;
}

void
Environment::
updatePrevContacts(Eigen::Vector2d handContacts)
{
	prevContact = curContact;

	if(handContacts[0]>0.5 || handContacts[1]>0.5)
		curContact = true;
	else
		curContact = false;
}



Eigen::Vector3d 
Environment::
computeBallPosition()
{
    double g = 9.81;
    // std::cout<<"criticalPoint_targetBallVelocity : "<<criticalPoint_targetBallVelocity.transpose()<<std::endl;
    Eigen::Vector3d cur_targetBallPosition = criticalPoint_targetBallPosition;
    // std::cout<<"cur_targetBallPosition : "<<cur_targetBallPosition.transpose()<<std::endl;
    double t = (curFrame - criticalPointFrame)/30.0;
    // std::cout<<"t : "<<t<<std::endl;
    cur_targetBallPosition += t*criticalPoint_targetBallVelocity;
    cur_targetBallPosition[1] += -g/2.0*pow(t,2);

    if(cur_targetBallPosition[1] < 0)
    {
        // double t1=0, t2=t;
        // double h = criticalPoint_targetBallPosition[1];
        // double v = criticalPoint_targetBallVelocity[1];
        // double up = t;
        // double down = 0;
        // while(abs(h + v*t1 - g/2.0*pow(t1,2))>1E-3)
        // {
        //     if(h + v*t1 - g/2.0*pow(t1,2) > 0)
        //     {
        //         down = t1;
        //         t1 = (t1+up)/2.0;
        //         t2 = t-t1;
        //     }
        //     else
        //     {
        //         up = t1;
        //         t1 = (t1+down)/2.0;
        //         t2 = t-t1;
        //     }
        // }

		double t1 = getBounceTime(criticalPoint_targetBallPosition[1], criticalPoint_targetBallVelocity[1], t);
		double t2 = t - t1;
        // std::cout<<"T : "<<t<<" "<<t1<<" "<<t2<<std::endl;

        Eigen::Vector3d bouncePosition = criticalPoint_targetBallPosition;
        bouncePosition += t1*criticalPoint_targetBallVelocity;
        bouncePosition[1] += -g/2.0*pow(t1,2);

        Eigen::Vector3d bouncedVelocity = criticalPoint_targetBallVelocity;
        bouncedVelocity[1] += -g*t1;
        bouncedVelocity *= 0.85;
        bouncedVelocity[1] *= -1;
        cur_targetBallPosition = bouncePosition + t2*bouncedVelocity;
        cur_targetBallPosition[1] += -g/2.0*pow(t2,2);

		if(cur_targetBallPosition[1] < 0)
		{
			double t1_ = getBounceTime(bouncePosition[1], bouncedVelocity[1], t-t1);

			cur_targetBallPosition = bouncePosition + bouncedVelocity*t1_;
			cur_targetBallPosition[1] = 0.0;

		}
    }
    return cur_targetBallPosition;
}

//upper bound for fast binary search
double getBounceTime(double startPosition, double startVelocity, double upperbound)
{
	double g= 9.81;
	double curPosition = startPosition + startVelocity*upperbound - g*0.5*pow(upperbound,2);
	assert(curPosition<0);
	double lowerBound = 0.0;
	if(startPosition <= 1E-3)
	{
		lowerBound = 0.1;
	}

	double t1=lowerBound;
	double t2=upperbound;
	double h = startPosition;
	double v = startVelocity;
	double up = t2;
	double down = t1;
	while(abs(h + v*t1 - g/2.0*pow(t1,2))>1E-3)
	{
		if(h + v*t1 - g/2.0*pow(t1,2) > 0)
		{
			down = t1;
			t1 = (t1+up)/2.0;
			t2 = upperbound-t1;
		}
		else
		{
			up = t1;
			t1 = (t1+down)/2.0;
			t2 = upperbound-t1;
		}
	}
	return t1;
}

// Eigen::Vector3d 
// Environment::
// getTargetBallGlobalPosition();
// {
// 	Motion::MotionSegment* ms = mMotionGenerator->motionGenerators[0]->mMotionSegment;
// 	Motion::Root root = ms->getLastPose()->getRoot();

// 	Eigen::Isometry3d baseToRoot = mMotionGenerator->getBaseToRootMatrix(root);


// 	// Get goalpost position
// 	Eigen::Vector3d relTargetPosition;
// 	relTargetPosition = baseToRoot.inverse()*mTargetBallPosition;

// 	state.resize(ICAPosition.rows() + relTargetPosition.rows());
// }


/*
bool isNotCloseToBall(Eigen::VectorXd curState)
{
	return curState.segment(_ID_BALL_P,2).norm() >= 0.15;
}

bool isNotCloseToBall_soft(Eigen::VectorXd curState)
{
	return curState.segment(_ID_BALL_P,2).norm() >= 0.25;
}


Eigen::VectorXd localStateToOriginState(Eigen::VectorXd localState, int mNumChars)
{
	Eigen::VectorXd originState = localState;
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
*/
