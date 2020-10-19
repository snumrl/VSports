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
#include "../vsports/common.h"

using namespace std;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::collision;
using namespace dart::constraint;

#define RESET_ADAPTING_FRAME 15
#define ACTION_SIZE 14
#define CONTROL_VECTOR_SIZE 14
#define NUM_ACTION_TYPE 5

// p.rows() + v.rows() + ballP.rows() + ballV.rows() +
// 	ballPossession.rows() + kickable.rows() + goalpostPositions.rows()

Environment::
Environment(int control_Hz, int simulation_Hz, int numChars, std::string bvh_path, std::string nn_path)
:mControlHz(control_Hz), mSimulationHz(simulation_Hz), mNumChars(numChars), mWorld(std::make_shared<dart::simulation::World>()),
mIsTerminalState(false), mTimeElapsed(0), mNumIterations(0), mSlowDuration(180), mNumBallTouch(0), endTime(15),
criticalPointFrame(0), curFrame(0), mIsFoulState(false), gotReward(false), violatedFrames(0)
{
	std::cout<<"Envionment Generation --- ";
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
	this->endTime = 10;
	this->initCharacters(bvh_path);
	// this->initMotionGenerator(nn_path);

	mNormalizer = new Normalizer("../extern/ICA/motions/"+nn_path+"/data/xNormal.dat", 
								"../extern/ICA/motions/"+nn_path+"/data/yNormal.dat");
}

void 
Environment::initialize(ICA::dart::MotionGeneratorBatch* mgb, int batchIndex)
{
	mMgb = mgb;
	mBatchIndex = batchIndex;

    mWorld->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));



	this->criticalPoint_targetBallPosition = Eigen::Vector3d(0.0, 0.85, 0.0);
	this->criticalPoint_targetBallVelocity = Eigen::Vector3d(0.0, 0.0, 0.0);
	criticalPointFrame = 0;
	curFrame = 0;
	this->slaveReset();

	// mPrevEnvSituations.resize(mNumChars);
	// for(int i=0;i<2;i++)
	// {
	// 	mPrevEnvSituations[i] = new EnvironmentPackage();
	// 	mPrevEnvSituations[i]->saveEnvironment(this);
	// 	// exit(0);
	// }

	mPrevPlayer= -1;
	// std::cout<<"Success"<<std::endl;



	actionNameMap[0] = "dribble";
	actionNameMap[1] = "pass";
	actionNameMap[2] = "receive";
	actionNameMap[3] = "shoot";
	actionNameMap[4] = "walk";
	actionNameMap[5] = "run";
	actionNameMap[6] = "pivot_l";
	actionNameMap[7] = "pivot_r";
}

void
Environment::
setPositionFromBVH(int index, int bvhFrame)
{
	BVHmanager::setPositionFromBVH(mCharacters[index]->getSkeleton(), mBvhParser, bvhFrame);
}

// void
// Environment::
// initMotionGenerator(std::string dataPath)
// {
// 	initDartNameIdMapping();
// 	mMotionGenerator = new ICA::dart::MotionGenerator(dataPath, this->dartNameIdMap);

// 	Eigen::VectorXd targetZeroVec(15);
// 	targetZeroVec.setZero();

// 	targetZeroVec[4+4] = 1;

// 	BVHmanager::setPositionFromBVH(mCharacters[0]->getSkeleton(), mBvhParser, 50);
// 	Eigen::VectorXd bvhPosition = mCharacters[0]->getSkeleton()->getPositions();

// 	for(int i=0;i<RESET_ADAPTING_FRAME;i++)
// 		mMotionGenerator->setCurrentPose(bvhPosition, Utils::toStdVec(dribbleDefaultVec));


// 	for(int i=0;i<RESET_ADAPTING_FRAME;i++)
// 		mMotionGenerator->generateNextPoseAndContacts(Utils::toStdVec(dribbleDefaultVec));

// 	// Eigen::VectorXd zeroAction = mActions[0];
// 	// zeroAction.setZero();
// 	auto nextPoseAndContacts = mMotionGenerator->generateNextPoseAndContacts(Utils::toStdVec(dribbleDefaultVec));
//     Eigen::VectorXd nextPosition = std::get<0>(nextPoseAndContacts);
//     mCharacters[0]->getSkeleton()->setPositions(nextPosition);
// 	curFrame++;
// }

std::map<std::string, int>
Environment::
initDartNameIdMapping()
{    
	int numFingerNodes = 4;
	SkeletonPtr bvhSkel = mCharacters[0]->getSkeleton();
	int curIndex = 0;
	for(int i=0;i<bvhSkel->getNumBodyNodes();i++)
	{
		if(bvhSkel->getBodyNode(i)->getName().find("Finger")!=std::string::npos)
			continue;
		this->dartNameIdMap[bvhSkel->getBodyNode(i)->getName()] = curIndex;
		curIndex += bvhSkel->getBodyNode(i)->getParentJoint()->getNumDofs();
	}
	return this->dartNameIdMap;
}




void Environment::addFingerSegmentToSkel(SkeletonPtr skel)
{
	double fingerLength = 0.07;
	Eigen::Isometry3d pb2j;
	Eigen::Isometry3d cb2j;

	pb2j.setIdentity();
	pb2j.translation() = Eigen::Vector3d(0.095, 0.0, 0.0);
	cb2j.setIdentity();
	cb2j.translation() = Eigen::Vector3d(-fingerLength/2.0, 0.0, 0.0);
	SkelMaker::makeWeldJointBody("RightFinger", skel, skel->getBodyNode("RightHand"), SHAPE_TYPE::BOX, Eigen::Vector3d(fingerLength, 0.03, 0.05),
	 pb2j, cb2j);

	// pb2j.setIdentity();
	// pb2j.translation() = Eigen::Vector3d(fingerLength/2.0, 0.0, 0.0);
	// cb2j.setIdentity();
	// cb2j.translation() = Eigen::Vector3d(0.0, -0.125, 0.0);
	// SkelMaker::makeRevoluteJointBody("RightFingerBall", skel, skel->getBodyNode("RightFinger"), SHAPE_TYPE::BALL, Eigen::Vector3d(0.02, 0.02, 0.02),
	//  pb2j, cb2j);


	pb2j.setIdentity();
	pb2j.translation() = Eigen::Vector3d(0.095, 0.0, 0.0);
	cb2j.setIdentity();
	cb2j.translation() = Eigen::Vector3d(-fingerLength/2.0, 0.0, 0.0);
	SkelMaker::makeWeldJointBody("LeftFinger", skel, skel->getBodyNode("LeftHand"), SHAPE_TYPE::BOX, Eigen::Vector3d(fingerLength, 0.03, 0.05),
	 pb2j, cb2j);

	// pb2j.setIdentity();
	// pb2j.translation() = Eigen::Vector3d(fingerLength/2.0, 0.0, 0.0);
	// cb2j.setIdentity();
	// cb2j.translation() = Eigen::Vector3d(0.0, -0.125, 0.0);
	// SkelMaker::makeRevoluteJointBody("LeftFingerBall", skel, skel->getBodyNode("LeftFinger"), SHAPE_TYPE::BALL, Eigen::Vector3d(0.02, 0.02, 0.02),
	//  pb2j, cb2j);


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
	addFingerSegmentToSkel(bvhSkel);
	// cout<<charNames[0]<<endl;

	// std::cout<<"LeftHand : "<<bvhSkel->getIndexOf(bvhSkel->getBodyNode("LeftHand"))<<std::endl;;
	// std::cout<<"LeftFinger : "<<bvhSkel->getIndexOf(bvhSkel->getBodyNode("LeftFinger"))<<std::endl;;
	// std::cout<<"LeftFingerBall : "<<bvhSkel->getIndexOf(bvhSkel->getBodyNode("LeftFingerBall"))<<std::endl;;


	// in the case of single player
	mCharacters[0]->mSkeleton = bvhSkel;


	BVHmanager::setPositionFromBVH(bvhSkel, mBvhParser, 100);
	mWorld->addSkeleton(bvhSkel);



	mActions.resize(mNumChars);
	mPrevActions.resize(mNumChars);
	for(int i=0;i<mNumChars;i++)
	{
		mActions[i].resize(ACTION_SIZE);
		mPrevActions[i].resize(ACTION_SIZE);

		mActions[i].setZero();
		mPrevActions[i].setZero();
	}

	mCurActionTypes.resize(mNumChars);
	mPrevActionTypes.resize(mNumChars);

	mStates.resize(mNumChars);
	mLocalStates.resize(mNumChars);
	mAccScore.resize(mNumChars);
	mAccScore.setZero();

	mPrevCOMs.resize(mNumChars);
	for(int i=0;i<mNumChars;i++)
	{
		mPrevCOMs[i] = mCharacters[i]->getSkeleton()->getRootBodyNode()->getCOM();
	}
	mCurCriticalActionTimes.resize(mNumChars);
	for(int i=0;i<mNumChars;i++)
	{
		mCurCriticalActionTimes[i] = 0;
	}
	mPrevBallPossessions.resize(mNumChars);
	mCurBallPossessions.resize(mNumChars);
	mDribbled.resize(mNumChars);
	for(int i=0;i<mNumChars;i++)
	{
		mPrevBallPossessions[i] = false;
		mCurBallPossessions[i] =false;
		mDribbled[i] = false;
	}

	prevContact.resize(mNumChars);
	curContact.resize(mNumChars);

	mLFootDetached.resize(mNumChars);
	mRFootDetached.resize(mNumChars);
	for(int i=0;i<mNumChars;i++)
	{
		prevContact[i] = -1;
		curContact[i] = -1;
		mLFootDetached[i] = false;
		mRFootDetached[i] = false;
	}
	bsm.resize(mNumChars);
	for(int i=0;i<mNumChars;i++)
	{
		bsm[i] = new BStateMachine();
	}
	mLFootContacting.resize(mNumChars);
	mRFootContacting.resize(mNumChars);
	mLLastFootPosition.resize(mNumChars);
	mRLastFootPosition.resize(mNumChars);

	for(int i=0;i<mNumChars;i++)
	{
		prevContact[i] = -1;
		curContact[i] = -1;

		mLFootContacting[i] = false;
		mRFootContacting[i] = false;

		mLLastFootPosition[i].setZero();
		mRLastFootPosition[i].setZero();


	}

	mActionGlobalBallPosition.resize(mNumChars);
	mActionGlobalBallVelocity.resize(mNumChars);

	mChangeContactIsActive.resize(mNumChars);

	for(int i=0;i<mNumChars;i++)
	{
		mChangeContactIsActive[i] = false;
	}
	
	dribbleDefaultVec.resize(CONTROL_VECTOR_SIZE);
	dribbleDefaultVec.setZero();
	//**action Type

	// dribbleDefaultVec.segment(0,2) = Eigen::Vector2d(30.0, 0.0);
	dribbleDefaultVec.segment(2,2) = Eigen::Vector2d(50.0, -50.0);
	dribbleDefaultVec[4] = 1.0;


	// initBehaviorTree();
}


void setSkelCollidable(SkeletonPtr skel, bool collidable = true)
{
	for(int i=0;i<skel->getNumBodyNodes();i++)
	{ 
		skel->getBodyNode(i)->setCollidable(collidable);
	}
}

Eigen::VectorXd
Environment::
getMGAction(int index)
{
	// std::cout<<"GET MG ACTION : "<<std::endl;
	return mActions[index].segment(0,CONTROL_VECTOR_SIZE);
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
	mPrevBallPosition = ballSkel->getCOM();
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

	// saveEnvironment();

	for(int i=0;i<mCharacters.size();i++)
	{
		applyAction(i);
	}
	// std::cout<<mPrevActions[0].transpose()<<std::endl;
	// std::cout<<mActions[0].transpose()<<std::endl;
	// std::cout<<std::endl;

	int sim_per_control = this->getSimulationHz()/this->getControlHz();
	for(int i=0;i<sim_per_control;i++)
	{
		this->step();
	}
	// std::copy(mActions.begin(), mActions.end(), mPrevActions.begin());
	resetCount--;
	if(resetCount < -1)
		resetCount = -1;

	curFrame++;

	// if(isTerminalState())
	// {
	// 	reset();
	// }

	// getRewards();
	// cout<<"Here?"<<endl;
	// mWindow->display();
	// cout<<"end"<<endl;
}

// finger : -0.4 ~ M_PI/2.0
// fingerBall : -1.3 ~ 0.2
double constrainedFingerAngle(double fingerAngle)
{
	if(fingerAngle < -0.4)
		return -0.4;
	if(fingerAngle > 1.0)
		return 1.0;
	return fingerAngle;
}

double constrainedFingerBallAngle(double fingerBallAngle)
{
	if(fingerBallAngle<-1.2)
		return -1.2;
	if(fingerBallAngle>0.2)
		return 0.2;
	return fingerBallAngle;
}

void
Environment::
stepAtOnce(std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool> nextPoseAndContacts)
{
	for(int index=0;index<mCharacters.size();index++)
	{
		// time_check_start();

		mPrevBallPosition = ballSkel->getCOM();
		for(int i=0;i<mNumChars;i++)
		{
			mPrevCOMs[i] = mCharacters[i]->getSkeleton()->getRootBodyNode()->getCOM();
			mPrevBallPossessions[i] = mCurBallPossessions[i];
		}


	    Eigen::VectorXd nextPosition = std::get<0>(nextPoseAndContacts);
	    // std::cout<<"Skel position : "<<std::endl;
	    // std::cout<<nextPosition.transpose()<<std::endl;
	    mCharacters[index]->mSkeleton->setPositions(nextPosition);
	    mCharacters[index]->mSkeleton->setVelocities(mCharacters[0]->mSkeleton->getVelocities().setZero());


	    Eigen::Vector4d nextContacts = std::get<1>(nextPoseAndContacts).segment(0,4);
		mCurBallPossessions[index] = std::get<2>(nextPoseAndContacts);
		if(mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3)
		{
			if(mCurCriticalActionTimes[index] >=0)
			{
				mCurBallPossessions[index] = true;
			}

			if(mCurCriticalActionTimes[index] < 0)
			{
				mCurBallPossessions[index] = false;
			}
		}
		if(mCurActionTypes[index] == 2)
		{

		}
		if(mCurActionTypes[index] == 4 || mCurActionTypes[index] == 5)
		{
			mCurBallPossessions[index] = false;
		}
		if(mCurActionTypes[index] == 0 || mCurActionTypes[index] == 6 || mCurActionTypes[index] == 7)
		{
			mCurBallPossessions[index] = true;
		}
	   	updatePrevContacts(index, nextContacts.segment(2,2));


	   	if(!mCurBallPossessions[index])
	   	{
	   		curContact[index] = -1;
	   	}
	   	else
	   	{
	   		if(mCurActionTypes[index] ==1 || mCurActionTypes[index] ==3 )
	   		{
	   			if(curContact[index] == -1)
		   		{
		   			curContact[index] = prevContact[index];
		   		}
	   		}
	   		
	   	}

		SkeletonPtr skel = mCharacters[index]->getSkeleton();


	    if(mPrevBallPossessions[index] && !mCurBallPossessions[index])
	    {
	        this->criticalPointFrame = curFrame-1;
	        this->criticalPoint_targetBallPosition = this->curBallPosition;

	        Eigen::Vector3d predictedBallPosition;

	        if(prevContact[index] == 0)
	        {
	        	predictedBallPosition = skel->getBodyNode("LeftHand")->getWorldTransform()*Eigen::Vector3d(0.14, 0.16, 0.0);
	        }
	        if(prevContact[index] == 1)
	        {
	        	predictedBallPosition = skel->getBodyNode("RightHand")->getWorldTransform()*Eigen::Vector3d(0.14, 0.16, 0.0);
	        }
	        if(prevContact[index] == 2)
	        {
	        	predictedBallPosition = skel->getBodyNode("LeftHand")->getWorldTransform()*Eigen::Vector3d(0.14, 0.16, 0.0)
				+skel->getBodyNode("RightHand")->getWorldTransform()*Eigen::Vector3d(0.14, 0.16, 0.0);
				predictedBallPosition /= 2.0;
	        }

			double speed= (predictedBallPosition - this->prevBallPositions[0]).norm()*15;

			speed /= 5.0;
            // std::cout<<"SPEED : "<<speed<<std::endl;
            // speed= sqrt(speed);
            if(speed > 1.3)
                speed = 1.3;
            if(speed < 0.5)
                speed = 0.5;

	        Eigen::Isometry3d headT = skel->getBodyNode("Head")->getWorldTransform();
	        
            Eigen::Vector3d ballDirection;

            if(mCurActionTypes[index] == 3)
                ballDirection = Eigen::Vector3d(0.7, 1.1, 0.0);
            else
                ballDirection = Eigen::Vector3d(0.0, 1.35, 0.0);

	        // this->criticalPoint_targetBallVelocity = headT.linear()*ballDirection*5*speed;

	        this->criticalPoint_targetBallVelocity = getRootT(index).linear()* (mActions[index].segment(4+NUM_ACTION_TYPE,3)/100.0);
	    }


	    //Update hand Contacts;

	    // {
	    	// if((mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3) 
	    	// 	&& mCurCriticalActionTimes[index]<10
	    	// 	&& !mChangeContactIsActive[index])
	    	// {
	    	// 	updatePrevContacts(index, Eigen::Vector2d::Zero());
	    	// 	mCurBallPossessions[index] = false;
	    	// }
	    	// else

	    // }

	    if((mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3) 
	    		&& mCurCriticalActionTimes[index]<-10)
	    {
	    	curContact[index] = -1;
	    }


	    if(mCurActionTypes[index] == 4 || mCurActionTypes[index] == 5)
	    {
	    	updatePrevBallPositions(computeBallPosition());
	    }
	    else if(curContact[index] >= 0)
	    {
	    	updatePrevBallPositions(computeHandBallPosition(index));
	    }
	    else if(mCurActionTypes[index] == 0)
	    {
			updatePrevBallPositions(std::get<1>(nextPoseAndContacts).segment(4,3));
	    }
	    else if(mCurBallPossessions[index])
	    {
	    	updatePrevBallPositions(std::get<1>(nextPoseAndContacts).segment(4,3));
	    }
	    else
	    {
	    	updatePrevBallPositions(computeBallPosition());
	    }


	    Eigen::Vector6d ballPosition;
	    ballPosition.setZero();

	    ballPosition.segment(3,3) = curBallPosition;

	    ballSkel->setPositions(ballPosition);

	    Eigen::Vector6d zeroVelocity;
	    zeroVelocity.setZero();
	    ballSkel->setVelocities(zeroVelocity);


	    mPrevActionTypes[index] = mCurActionTypes[index];


	    //** get foot contacting
		if(std::get<1>(nextPoseAndContacts)[0] > 0.5)
		{
			if(!mLFootContacting[index])
				mLLastFootPosition[index] = skel->getBodyNode("LeftToe")->getWorldTransform().translation();
			mLFootContacting[index] = true;
		}
		else
		{
			mLFootContacting[index] = false;
		}

		if(std::get<1>(nextPoseAndContacts)[1] > 0.5)
		{
			if(!mRFootContacting[index])
				mRLastFootPosition[index] = skel->getBodyNode("RightToe")->getWorldTransform().translation();
			mRFootContacting[index] = true;
		}
		else
		{
			mRFootContacting[index] = false;
		}


		if(mLFootContacting[index])
		{
			Eigen::Vector3d curLFootPosition = skel->getBodyNode("LeftToe")->getWorldTransform().translation();
			Eigen::Vector3d footDiff = curLFootPosition - mLLastFootPosition[index];
			footDiff[1] = 0.0;
			// std::cout<<(curLFootPosition - mLLastFootPosition[index]).norm()<<std::endl;
			if(footDiff.norm()>0.50)
			{
				// std::cout<<"Left Foot Sliding"<<std::endl;
				// mIsTerminalState = true;
			}
		}

		if(mRFootContacting[index])
		{
			Eigen::Vector3d curRFootPosition = skel->getBodyNode("RightToe")->getWorldTransform().translation();
			Eigen::Vector3d footDiff = curRFootPosition - mRLastFootPosition[index];
			footDiff[1] = 0.0;
			if(footDiff.norm()>0.50)
			{
				// std::cout<<"Right Foot Sliding"<<std::endl;
				// mIsTerminalState = true;
			}
		}


		// time_check_end();
	}


	if(!ballGravityChecker(0))
	{
		// bsm[index]->prevAction = 4;
		// bsm[index]->curState = BasketballState::POSITIONING;
		// bsm[index]->prevAction = 0;
		// bsm[index]->curState = BasketballState::DRIBBLING;
		mIsTerminalState = true;

		std::cout<<"Ball floating"<<std::endl;
	}


	int sim_per_control = this->getSimulationHz()/this->getControlHz();
	for(int i=0;i<sim_per_control;i++)
	{
		this->step();
	}
	resetCount--;
	if(resetCount < -1)
		resetCount = -1;
	curFrame++;
}



Eigen::VectorXd
Environment::
getState(int index)
{
	Eigen::VectorXd state;
	// state.setZero();

	// Use the same format of the motion learning

	std::vector<double> _ICAPosition;
	Motion::MotionSegment* ms = mMgb->motionGenerators[0]->mMotionSegments[mBatchIndex];
    MotionRepresentation::getData(ms, _ICAPosition, ms->mPoses.size()-1);
	Eigen::VectorXd ICAPosition = Utils::toEigenVec(_ICAPosition);

	Eigen::Vector4d contacts = ICAPosition.segment(0,4);

	contacts.segment(2,2).setZero();
	if(curContact[index] == 0 || curContact[index] == 2)
	{
		contacts[2] = 1.0;
	}
	if(curContact[index] == 1 || curContact[index] == 2)
	{
		contacts[3] = 1.0;
	}

	// Motion::Root root = ms->getLastPose()->getRoot();

	// Eigen::Isometry3d baseToRoot = ICA::dart::getBaseToRootMatrix(root);

	// ICAPosition.segment(MotionRepresentation::posOffset,3) = relCurBallPosition;


	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	Eigen::Isometry3d rootT = getRootT(index);

	std::vector<std::string> EEJoints;
	EEJoints.push_back("LeftHand");
	EEJoints.push_back("RightHand");

	// EEJoints.push_back("LeftFinger");
	// EEJoints.push_back("RightFinger");

	// EEJoints.push_back("LeftFingerBall");
	// EEJoints.push_back("RightFingerBall");


	EEJoints.push_back("LeftToe");
	EEJoints.push_back("RightToe");

	EEJoints.push_back("LeftFoot");
	EEJoints.push_back("RightFoot");
	Eigen::VectorXd skelPosition(skel->getNumDofs() + 3*EEJoints.size());

	skelPosition.segment(0, skel->getNumDofs()) = skel->getPositions();

	for(int i=0;i<EEJoints.size();i++)
	{
		skelPosition.segment(skel->getNumDofs()+3*i, 3) = rootT.inverse() * 
			skel->getBodyNode(EEJoints[i])->getWorldTransform().translation();
	}


	Eigen::Vector3d relCurBallPosition = rootT.inverse()*curBallPosition;

	// Get goalpost position
	Eigen::Vector3d relTargetPosition;
	relTargetPosition = rootT.inverse()*mTargetBallPosition;


	Eigen::Vector3d relBallToTargetPosition;
	relBallToTargetPosition = (relTargetPosition - relCurBallPosition);


	// ICAPosition.segment(5,3) /= 100.0;
	// ICAPosition.segment(8,3) /= 100.0;
	// ICAPosition.segment(8+3+8+1, 22*3) /= 100.0;


	// std::cout<<"##############"<<std::endl;
	// std::cout<<mTargetBallPosition<<std::endl;
	// std::cout<<relTargetPosition<<std::endl;
	// std::cout<<"relTargetPosition : "<<relTargetPosition.transpose()<<std::endl;

	Eigen::Vector6d goalpostPositions;
	goalpostPositions.segment(0,3) = Eigen::Vector3d(14.0 -1.5 + 0.05, 3.1+0.2, 0.0);
	goalpostPositions.segment(3,3) = Eigen::Vector3d(-(14.0 -1.5 + 0.05), 3.1+0.2, 0.0);


	goalpostPositions.segment(0,3) = rootT.inverse() * ((Eigen::Vector3d) goalpostPositions.segment(0,3));
	goalpostPositions.segment(3,3) = rootT.inverse() * ((Eigen::Vector3d) goalpostPositions.segment(3,3));

	// goalpostPositions/= 100.0;

	Eigen::VectorXd curActionType(5);
	curActionType.setZero();
	// for(int i=0;i<curActionType.size();i++)
	// {
	// 	if(mActions[0][4+i] > 0.5)
	// 		curActionType[i] = 1;
	// }

	curActionType[mCurActionTypes[index]] = 1;
	// assert(curActionType.norm()==1);

	// std::cout<<"BallPossession : "<<ICAPosition[8+3+8]<<std::endl;


	Eigen::Vector3d ballVelocity;
	if(!mPrevBallPossessions[index] && mCurBallPossessions[index])
	{
		ballVelocity.setZero();
		curBallVelocity.setZero();
	}
	else
	{	
		curBallVelocity = 30.0 * (curBallPosition - prevBallPositions[1]);
		if(curBallVelocity.norm() > 20.0)
			curBallVelocity.setZero();
		ballVelocity = rootT.linear().inverse()* curBallVelocity;
		// std::cout<<"A : "<<curBallPosition.transpose()<<std::endl;
		// std::cout<<"B : "<<prevBallPositions[1].transpose()<<std::endl;
	}

	// std::cout<<"Cur ball velocity : "<<curBallVelocity.transpose()<<std::endl;



	// std::cout<<"cur ball velocity : "<<curBallVelocity.transpose()<<std::endl;

	// ballVelocity/=100.0;

	// ballVelocity/=4.0;
	Eigen::Vector4d curSMState;
	curSMState.setZero();
	curSMState[bsm[index]->getStateWithInt()] = 1.0;

	Eigen::VectorXd availableActions(5);
	availableActions.setZero();
	std::vector<int> availableActionList = bsm[index]->getAvailableActions();
	for(int i=0;i<availableActionList.size();i++)
	{
		if(availableActionList[i] < 5)
			availableActions[availableActionList[i]] = 1;
	}

	std::vector<Eigen::Vector3d> relObstacles(mObstacles.size());
	for(int i=0;i<mObstacles.size();i++)
	{
		relObstacles[i] = rootT.inverse()*mObstacles[i];
		// relObstacles[i][1] = 0;
	}


	mCharacters[index]->blocked = false;
	for(int i=0;i<relObstacles.size();i++)
	{
		Eigen::Vector3d projectedGoalpostPositions = goalpostPositions.segment(0,3);
		projectedGoalpostPositions[1] = 0.0;
		Eigen::Vector3d temp = projectedGoalpostPositions.normalized() * relObstacles[i].dot(projectedGoalpostPositions.normalized());
		// std::cout<<"temp.norm() : "<<temp.norm()<<std::endl;
		// std::cout<<"projectedGoalpostPositions.norm() : "<<projectedGoalpostPositions.norm()<<std::endl;
		if(temp.norm() > projectedGoalpostPositions.norm())
			continue;
		temp[1] = relObstacles[i][1];
		// std::cout<<"Temp : "<<temp.transpose()<<std::endl;
		// std::cout<<"relObstacles : "<<relObstacles[i].transpose()<<std::endl;
		double distance = (relObstacles[i] - temp).norm();
		// std::cout<<distance<<std::endl;
		if(distance < 0.5)
		{
			mCharacters[index]->blocked = true;
			break;
		}
	}




	// std::cout<<" Cur state is "<<bsm[index]->curState<<std::endl;
	// std::cout<<availableActions.transpose()<<std::endl;

	// std::cout<<"getState :"<<std::endl;
	// std::cout<<"skelPosition.transpose(): "<<skelPosition.transpose()<<std::endl;
	// std::cout<<"relCurBallPosition.transpose(): "<<relCurBallPosition.transpose()<<std::endl;
	// std::cout<<"relTargetPosition.transpose(): "<<relTargetPosition.transpose()<<std::endl;
	// std::cout<<"relBallToTargetPosition.transpose(): "<<relBallToTargetPosition.transpose()<<std::endl;
	// std::cout<<"goalpostPositions.transpose(): "<<goalpostPositions.transpose()<<std::endl;
	// std::cout<<"contacts.transpose(): "<<contacts.transpose()<<std::endl;

	state.resize(skelPosition.rows() + relCurBallPosition.rows() + relTargetPosition.rows() + relBallToTargetPosition.rows() + goalpostPositions.rows() 
		+ contacts.rows() + 1 + 1 + 3 +curActionType.rows()+curSMState.rows()+availableActions.rows() + relObstacles.size()*3 + 1);
	 //+ ballVelocity.rows()+2+curActionType.rows());
	
	int curIndex = 0;
	for(int i=0;i<skelPosition.rows();i++)
	{
		state[curIndex] = skelPosition[i];
		curIndex++;
	}
	for(int i=0;i<relCurBallPosition.rows();i++)
	{
		state[curIndex] = relCurBallPosition[i];
		curIndex++;
	}
	for(int i=0;i<relTargetPosition.rows();i++)
	{
		state[curIndex] = relTargetPosition[i];
		curIndex++;
	}
	for(int i=0;i<relBallToTargetPosition.rows();i++)
	{
		state[curIndex] = relBallToTargetPosition[i];
		curIndex++;
	}
	for(int i=0;i<goalpostPositions.rows();i++)
	{
		state[curIndex] = goalpostPositions[i];
		curIndex++;
	}
	for(int i=0;i<contacts.rows();i++)
	{
		state[curIndex] = contacts[i];
		curIndex++;
	}

	state[curIndex] = mCurBallPossessions[index];
	curIndex++;

	state[curIndex]=mCurCriticalActionTimes[index]/30.0;
	curIndex++;

	for(int i=0;i<ballVelocity.rows();i++)
	{
		state[curIndex] = ballVelocity[i];
		curIndex++;
	}
	// state[curIndex] = mDribbled[index];
	// curIndex++;
	// state[curIndex] = mPrevPlayer == index? 1 : 0;
	// curIndex++;


	// // std::cout<<"Critical action time : "<<state[curIndex-1]<<std::endl;

	for(int i=0;i<curActionType.rows();i++)
	{
		state[curIndex] = curActionType[i];
		curIndex++;
	}
	for(int i=0;i<curSMState.rows();i++)
	{
		state[curIndex] = curSMState[i];
		curIndex++;
	}
	for(int i=0;i<availableActions.rows();i++)
	{
		state[curIndex] = availableActions[i];
		curIndex++;
	}

	for(int i=0;i<relObstacles.size();i++)
	{
		state[curIndex] = relObstacles[i][0];
		curIndex++;
		state[curIndex] = relObstacles[i][1];
		curIndex++;
		state[curIndex] = relObstacles[i][2];
		curIndex++;
	}
	state[curIndex] = mCharacters[index]->blocked;
	curIndex++;


	// std::cout<<std::setprecision(3);
	// std::cout<<state.segment(skelPosition.size(), 20).transpose()<<std::endl;
	// std::cout<<state.segment(skelPosition.size() + 20, 4).transpose()<<std::endl;
	// std::cout<<state.segment(skelPosition.size() + 24, state.size() - skelPosition.size()- 24).transpose()<<std::endl;
	// std::cout<<state.segment(skelPosition.size() +40, state.size()-40).transpose()<<std::endl;
	// std::cout<<std::endl;

	mStates[index] = state;
	// cout<<"getState end"<<endl;
	return state;
}

// Eigen::VectorXd
// Environment::
// getNormalizedState(int index)
// {
// 	// std::cout<<mNormalizer->normalizeState(getState(index)).transpose()<<std::endl;
// 	return mNormalizer->normalizeState(getState(index));
// 	// return getState(index);
// }

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
	double g= -9.81;

	bool fastTermination = true;
	// activates when fastTermination is on
	bool fastViewTermination = true;

	bool isDribble = true;

	if(isDribble)
	{
		if(!mCurBallPossessions[index])
		{
			mIsTerminalState = true;
			return 0;
		}

		// Dribble Direction Reward
		Eigen::Vector3d targetPlaneNormal = mTargetBallPosition - mCharacters[index]->getSkeleton()->getRootBodyNode()->getCOM();
		targetPlaneNormal[1] = 0.0;

		Eigen::Vector3d comTargetDirection = targetPlaneNormal.normalized();

		Eigen::Vector3d curRootVelocity = mCharacters[index]->getSkeleton()->getRootBodyNode()->getCOM() - mPrevCOMs[index];

		reward += 0.1*comTargetDirection.dot(curRootVelocity);

		if(targetPlaneNormal.norm() < 0.5)
		{
			reward = 1.0;
			mIsTerminalState = true;
		}

		return reward;

	}

	else
	{
		if(mCharacters[index]->blocked && mCurActionTypes[index] == 0)
			reward += 0.001;
		// Shoot Reward
		if(gotReward)
			return 0;


		if(!mCurBallPossessions[index])
		{
			if(fastViewTermination)
				mIsTerminalState = true;
			gotReward = true;

			Eigen::Vector3d relTargetPosition = mTargetBallPosition - criticalPoint_targetBallPosition;

			double h = relTargetPosition[1];

			double v = criticalPoint_targetBallVelocity[1];
			reward += 0.02*v*relTargetPosition.normalized().dot(criticalPoint_targetBallVelocity.normalized());

			//vt + 1/2 gt^2 = h
			// std::cout<<"v*v+2*g*h "<<v*v+2*g*h<<std::endl;
			if(v*v+2*g*h<0)
			{
				return reward;
			}
			double t = (-v -sqrt(v*v+2*g*h))/g;

			Eigen::Vector3d ballPositionOnThePlane = criticalPoint_targetBallPosition + criticalPoint_targetBallVelocity*t;
			ballPositionOnThePlane[1] = 0.0;

			Eigen::Vector3d targetPositionOnThePlane = mTargetBallPosition;
			targetPositionOnThePlane[1] = 0.0;

			if(!mCharacters[index]->blocked)
				reward += exp(0.3 * -pow((targetPositionOnThePlane - ballPositionOnThePlane).norm(),2));
			else
				reward += 0.2 * exp(0.3 * -pow((targetPositionOnThePlane - ballPositionOnThePlane).norm(),2));

			return reward;
		}
		else
		{
			return reward;
		}
	}

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
		// std::cout<<"Reward : "<<mAccScore[i]<<std::endl;

	}

	return rewards;
}

bool
Environment::
isCriticalAction(int actionType)
{
	if(actionType == 1 || actionType == 2 || actionType == 3)
		return true;
	else
		return false;
}


bool
Environment::
ballGravityChecker(int index)
{
	double targetMinHeight = 0.3;
	if(curContact[index] != -1)
	{
		prevFreeBallPositions.clear();
		violatedFrames = 0;
		return true;
	}
	else
	{
		if(curBallPosition[1]<targetMinHeight)
		{
			prevFreeBallPositions.clear();
			violatedFrames = 0;
			return true;
		}
		// std::cout<<"prevFreeBallPositions.size() :"<<prevFreeBallPositions.size()<<std::endl;
		prevFreeBallPositions.insert(prevFreeBallPositions.begin(),curBallPosition[1]);
		if(prevFreeBallPositions.size()>=4)
		{
			double prevBallVelocity = prevFreeBallPositions[1] - prevFreeBallPositions[3];
			double curBallVelocity = prevFreeBallPositions[0] - prevFreeBallPositions[2];
			if(curBallVelocity >= prevBallVelocity)
			{
				violatedFrames++;
				// std::cout<<"BALL ignored gravity"<<std::endl;
				// return false;
			}
			if(violatedFrames>2)
			{
				return false;
			}
		}
	}
	violatedFrames = 0;
	return true;

	// curContact[index]
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
	// std::cout<<a.transpose()<<std::endl;
	bool isNanOccured = false;

	for(int i=0;i<a.size();i++)
	{
		if(std::isnan(a[i]))
		{
			isNanOccured = true;
			// exit(0);
		}
	}
	mPrevActions = mActions;

	if(!isNanOccured)
	{
		mActions[index].setZero();
		mActions[index].segment(0,4) = a.segment(0,4);
		mActions[index].segment(4+NUM_ACTION_TYPE,5) = a.segment(4,5);
	}
	else
	{
		std::cout<<"Nan Action"<<std::endl;
		mActions[index].setZero();
		// mActions[index][4+4] = 1.0;
	}

	// int curActionType = 0;
 //    int maxIndex = 0;
 //    double maxValue = -100;
 //    for(int i=4;i<4+NUM_ACTION_TYPE;i++)
 //    {
 //        if(mActions[index][i] > maxValue)
 //        {
 //            maxValue = mActions[index][i];
 //            maxIndex = i;
 //        }
 //    }
 //    // std::cout<<"Original action : "<< maxIndex-4<<std::endl;

 //    curActionType = maxIndex-4;

 //    if(resetCount<0)
 //    	curActionType = 3;
 //    else
 //    	curActionType = 0;


    // std::cout<<"resetCount "<<resetCount<<std::endl;
    // std::cout<<"before bsm ActionType " <<curActionType<<std::endl;

    // if(isCriticalAction(mPrevActionTypes[index]))
    // {
    // 	if(mCurCriticalActionTimes[index] > -15)
    // 		curActionType = mPrevActionTypes[index];
    // 	else
    // 	{
    // 		bsm[index]->transition(mPrevActionTypes[index], true);
	   //  	if(bsm[index]->isAvailable(curActionType))
	   //  	{
	   //  		if(isCriticalAction(curActionType))
	   //  			curActionType = bsm[index]->transition(curActionType);
	   //  		else
	   //  			curActionType = bsm[index]->transition(curActionType, true);
	   //  	}
	   //  	else
	   //  		curActionType = bsm[index]->transition(curActionType);    	
	   //  }
    // }
    // else
    // {
    // 	if(bsm[index]->isAvailable(curActionType))
    // 	{
    // 		if(isCriticalAction(curActionType))
    // 			curActionType = bsm[index]->transition(curActionType);
    // 		else
    // 			curActionType = bsm[index]->transition(curActionType, true);
    // 	}
    // 	else
    // 		curActionType = bsm[index]->transition(curActionType);
    // }

    // std::cout<<"Set Action actiontype : "<<curActionType<<std::endl;
    // mCurActionTypes[index] = curActionType;
	// mActions[index].segment(4,NUM_ACTION_TYPE).setZero();

    mActions[index][4+mCurActionTypes[index]] = 1.0;


    if(mCurActionTypes[index] == 4 || mCurActionTypes[index] == 5)
    {
    	mActions[index].segment(2,2).setZero();
    }
    else
    {

    }

    if(mActions[index].segment(0,2).norm() > 150.0)
    {
    	mActions[index].segment(0,2) *= 150.0/mActions[index].segment(0,2).norm();
    }

    // if(mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3)
    // {
    // 	if(mActions[index].segment(0,2).norm() > 150.0)
	   //  {
	   //  	mActions[index].segment(0,2) *= 150.0/mActions[index].segment(0,2).norm();
	   //  }
    // }


    if(mActions[index].segment(2,2).norm() > 100.0)
    {
    	mActions[index].segment(2,2) *= 100.0/mActions[index].segment(2,2).norm();
    }

    // if(mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3)
    // {
    // 	if(mActions[index].segment(2,2).norm() > 100.0)
	   //  {
	   //  	mActions[index].segment(2,2) *= 100.0/mActions[index].segment(2,2).norm();
	   //  }
    // }



    if(mActions[index].segment(4+NUM_ACTION_TYPE,3).norm()>1300.0)
    {
    	mActions[index].segment(4+NUM_ACTION_TYPE,3) *= 1300.0/mActions[index].segment(4+NUM_ACTION_TYPE,3).norm();
    }

    if(mActions[index][4+NUM_ACTION_TYPE+3] > 250.0)
    {
    	mActions[index][4+NUM_ACTION_TYPE+3]  = 250.0;
    }
    else if(mActions[index][4+NUM_ACTION_TYPE+3] < 50.0)
    {
    	mActions[index][4+NUM_ACTION_TYPE+3] = 50.0;
    }

    if(!isCriticalAction(mCurActionTypes[index]))
    {
    	mActions[index].segment(4+NUM_ACTION_TYPE, mActions[index].rows()-(4+NUM_ACTION_TYPE)).setZero();
    }

	computeCriticalActionTimes();
/*
    if(mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3)
    {
	   	for(int i=0;i<2;i++)
	   		mActions[index][4+NUM_ACTION_TYPE+5+i] = mActions[index][4+NUM_ACTION_TYPE+5+i] >= 0.0 ? 1.0 : 0.0;

	   	for(int i=2;i<6;i++)
	   	{
	   		if(abs(mActions[index][4+NUM_ACTION_TYPE+5+i])>0.2)
	   		{
	   			double posneg = (mActions[index][4+NUM_ACTION_TYPE+5+i] > 0) - (mActions[index][4+NUM_ACTION_TYPE+5+i] < 0);
	   			mActions[index][4+NUM_ACTION_TYPE+5+i] = 0.2 * posneg;
	   		}
	   	}

	}*/


}


int
Environment::
setActionType(int index, int actionType)
{
	// if(actionType == 1)
	// 	std::cout<<"Action Type : "<<actionType<<std::endl;
	actionType *= 3;

	int curActionType = actionType;
	if(resetCount>=0)
		curActionType = 0;
	else if(actionType != 3)
		curActionType = 0;
	// curActionType = 0;

    if(isCriticalAction(mPrevActionTypes[index]))
    {
    	if(mCurCriticalActionTimes[index] > -10)
    		curActionType = mPrevActionTypes[index];
    	else
    	{
    		bsm[index]->transition(mPrevActionTypes[index], true);
	    	if(bsm[index]->isAvailable(curActionType))
	    	{
	    		if(isCriticalAction(curActionType))
	    			curActionType = bsm[index]->transition(curActionType);
	    		else
	    			curActionType = bsm[index]->transition(curActionType, true);
	    	}
	    	else
	    		curActionType = bsm[index]->transition(curActionType);    	
	    }
    }
    else
    {
    	if(bsm[index]->isAvailable(curActionType))
    	{
    		if(isCriticalAction(curActionType))
    			curActionType = bsm[index]->transition(curActionType);
    		else
    			curActionType = bsm[index]->transition(curActionType, true);
    	}
    	else
    		curActionType = bsm[index]->transition(curActionType);
    }
    // std::cout<<"Set Action actiontype : "<<curActionType<<std::endl;
    // bsm[index]->transition(curActionType);
    mCurActionTypes[index] = curActionType;

    return curActionType/3;
}

bool
Environment::
isTerminalState()
{
	if(resetCount>=0)
	{
		mIsTerminalState = false;
		return false;
	}
	if(mTimeElapsed > endTime)
	{
		mIsTerminalState = true;
	}
	if(mCurCriticalActionTimes[0] < -60)
		mIsTerminalState= true;

	// if((mCharacters[0]->getSkeleton()->getCOM()-mTargetBallPosition).norm() > 20.0)
	// 	mIsTerminalState = true;

	if(abs(mCharacters[0]->getSkeleton()->getCOM()[2])>15.0*0.5*1.3 || 
		mCharacters[0]->getSkeleton()->getCOM()[0]>28.0*0.5*1.3 || mCharacters[0]->getSkeleton()->getCOM()[0] < -4.0)
	{
		mIsTerminalState = true;

	}

	return mIsTerminalState;
}

bool
Environment::
isFoulState()
{
	return mIsFoulState;
}

void
Environment::
reset()
{
	mIsTerminalState = false;
	mIsFoulState = false;
	mTimeElapsed = 0;

	mAccScore.setZero();
	mNumBallTouch = 0;

	// mMotionGenerator->clear();
	resetCharacterPositions();
	resetTargetBallPosition();
	for(int i=0;i<mNumChars;i++)
	{
		mActions[i].setZero();
		mPrevActions[i].setZero();
		// mActions[i][4+4] = 1.0;
		// mPrevActions[i][4+4] = 1.0;

		// mCurActionTypes[i] = 4;
		// mPrevActionTypes[i] = 4;
		int curDefaultActionType = 3;

		mActions[i][4+curDefaultActionType] = 1.0;
		mPrevActions[i][4+0] = 1.0;

		mCurActionTypes[i] = curDefaultActionType;
		mPrevActionTypes[i] = 0;

		mCurCriticalActionTimes[i] = 30;
		mChangeContactIsActive[i] = false;

		curContact[i] = -1;
		prevFreeBallPositions.clear();
	}
	gotReward = false;
}

void
Environment::
slaveReset()
{
	resetCount = 45;
	mIsTerminalState = false;
	mIsFoulState = false;
	mTimeElapsed = 0;

	mAccScore.setZero();
	mNumBallTouch = 0;
	mMgb->clear(mBatchIndex);

	// std::cout<<"0000000000"<<std::endl;
	slaveResetCharacterPositions();
	// std::cout<<"1111111111"<<std::endl;
	slaveResetTargetBallPosition();
	// std::cout<<"2222222222"<<std::endl;

	mObstacles.clear();
	for(int i=0;i<0;i++)
		genObstacleNearGoalpost();

	for(int i=0;i<mNumChars;i++)
	{
		mActions[i].setZero();
		mPrevActions[i].setZero();
		// mActions[i][4+4] = 1.0;
		// mPrevActions[i][4+4] = 1.0;

		// mCurActionTypes[i] = 4;
		// mPrevActionTypes[i] = 4;
		int curDefaultActionType = 0;

		mActions[i][4+curDefaultActionType] = 1.0;
		mPrevActions[i][4+0] = 1.0;

		mCurActionTypes[i] = curDefaultActionType;
		mPrevActionTypes[i] = 0;

		mCurCriticalActionTimes[i] = 30;
		mChangeContactIsActive[i] = false;

		curContact[i] = -1;
		prevFreeBallPositions.clear();
	}
	gotReward = false;
}

void
Environment::
slaveResetTargetBallPosition()
{
	resetTargetBallPosition();
}

void
Environment::
slaveResetCharacterPositions()
{
	bool useHalfCourt = true;
	double xRange = 28.0*0.5*0.8;
	double zRange = 15.0*0.5*0.8;

	for(int i=0;i<mNumChars;i++)
	{
		mCharacters[i]->curLeftFingerPosition = 0.0;
		mCharacters[i]->curRightFingerPosition = 0.0;
		mCharacters[i]->curLeftFingerBallPosition = 0.0;
		mCharacters[i]->curRightFingerBallPosition = 0.0;
	}

	setPositionFromBVH(0, rand()%60+100);
	Eigen::VectorXd standPosition = mCharacters[0]->getSkeleton()->getPositions();


	standPosition[4] = 0.895;

	if(useHalfCourt)
	{
		double r = (double) rand()/RAND_MAX * zRange * 0.8 + zRange*0.2;
		double theta = (double) rand()/RAND_MAX * M_PI;


		standPosition[3] = xRange - r * sin(theta);
		standPosition[5] = 0 + r * cos(theta);
	}
	else
	{
		standPosition[3] = (double) rand()/RAND_MAX * xRange*2.0 - xRange;
		standPosition[5] = (double) rand()/RAND_MAX * zRange*2.0 - zRange;
	}

	Eigen::Vector3d curRootOrientation = standPosition.segment(0,3);

	double angle = curRootOrientation.norm();
	Eigen::Vector3d axis = curRootOrientation.normalized();

	Eigen::Vector3d curDirection = Eigen::AngleAxisd(angle, axis) * Eigen::Vector3d::UnitY();
	curDirection[1] = 0.0;

	curDirection.normalize();

	Eigen::AngleAxisd aa(angle, axis);
	Eigen::Vector3d goalDirection = Eigen::Vector3d(14.0 -1.5 + 0.05, 3.1+0.2, 0.0)- standPosition.segment(3,3);
	goalDirection[1] = 0.0;
	goalDirection.normalize();
	// std::cout<<curDirection.transpose()<<std::endl;
	// std::cout<<goalDirection.transpose()<<std::endl;

	if(curDirection.dot(goalDirection) <0.0)
	{
		// std::cout<<"Reverse"<<std::endl;
		aa = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY())*aa;
	}

	double randomDirection = rand()/(double)RAND_MAX * 2.0 * M_PI;

	bool directionToGoal = true;
	if(directionToGoal)
	{
		aa = Eigen::AngleAxisd(atan2(goalDirection[0], goalDirection[2])+M_PI/2.0 + randomDirection, Eigen::Vector3d::UnitY()) * Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitZ());
	}

	Eigen::Vector3d newRootOrientation = aa.angle() * aa.axis();
	// newRootOrientation.setZero();

	standPosition.segment(0,3) = newRootOrientation;




	criticalPoint_targetBallPosition = curBallPosition;
	criticalPoint_targetBallVelocity.setZero();


	if((double) rand()/RAND_MAX > 0.5)
		dribbleDefaultVec[3] *= -1;

	slaveResetTargetVector = dribbleDefaultVec;

	mMgb->clear(mBatchIndex);
	// std::cout<<"standPosition.transpose() "<<standPosition.transpose()<<std::endl;
	slaveResetStateVector = mMgb->dartPositionToCombinedPosition(standPosition, mBatchIndex);
	slaveResetPositionVector = standPosition;


	criticalPoint_targetBallPosition = curBallPosition;
	criticalPoint_targetBallVelocity.setZero();

	// if((double) rand()/RAND_MAX > 0.5)
	// 	dribbleDefaultVec[3] *= -1;

	// Eigen::VectorXd zeroAction = mActions[0];
	// zeroAction.setZero();
	// zeroAction[4+4] = 1;
	// auto nextPoseAndContacts = mMotionGenerator->generateNextPoseAndContacts(Utils::toStdVec(dribbleDefaultVec));
	curBallPosition.setZero();

    mCharacters[0]->getSkeleton()->setPositions(standPosition);
	for(int i=0;i<3;i++)
	{
		updatePrevBallPositions(curBallPosition);
	}

	Eigen::Vector6d ballPosition = ballSkel->getPositions();
	ballPosition.setZero();
	ballPosition.segment(3,3) = curBallPosition;
	ballSkel->setPositions(ballPosition);

	for(int i=0;i<mNumChars;i++)
	{
		mPrevCOMs[i] = mCharacters[i]->getSkeleton()->getRootBodyNode()->getCOM();
	}
	mPrevBallPosition = ballSkel->getCOM();

	for(int i=0;i<mNumChars;i++)
	{
		mPrevBallPossessions[i] = false;
		mCurBallPossessions[i] =false;
		mDribbled[i] = false;

		mLFootDetached[i] = false;
		mRFootDetached[i] = false;
		// bsm[i]->curState = BasketballState::POSITIONING;
		// bsm[i]->prevAction = 4;


		bsm[i]->curState = BasketballState::DRIBBLING;
		bsm[i]->prevAction = 0;

		mLFootContacting[i] = false;
		mRFootContacting[i] = false;

		mLLastFootPosition[i].setZero();
		mRLastFootPosition[i].setZero();

	}

	curFrame = 0;
	criticalPointFrame = 0; 
}

void 
Environment::
resetCharacterPositions()
{

	// std::cout<<"A"<<std::endl;
	bool useHalfCourt = true;
	double xRange = 28.0*0.5*0.8;
	double zRange = 15.0*0.5*0.8;	



	Eigen::VectorXd standPosition = mCharacters[0]->getSkeleton()->getPositions();
	standPosition[4] = 0.895;

	if(useHalfCourt)
	{
		double r = (double) rand()/RAND_MAX * zRange * 0.8 + zRange*0.2;
		double theta = (double) rand()/RAND_MAX * M_PI;
		// standPosition[3] = (double) rand()/RAND_MAX * xRange*1.0;
		// standPosition[5] = (double) rand()/RAND_MAX * zRange*1.0;

		standPosition[3] = xRange - r * sin(theta);
		standPosition[5] = 0 + r * cos(theta);
	}
	else
	{
		standPosition[3] = (double) rand()/RAND_MAX * xRange*2.0 - xRange;
		standPosition[5] = (double) rand()/RAND_MAX * zRange*2.0 - zRange;
	}

	Eigen::Vector3d curRootOrientation = standPosition.segment(0,3);

	double angle = curRootOrientation.norm();
	Eigen::Vector3d axis = curRootOrientation.normalized();

	Eigen::Vector3d curDirection = Eigen::AngleAxisd(angle, axis) * Eigen::Vector3d::UnitY();
	curDirection[1] = 0.0;
	curDirection.normalize();
	// std::cout<<curDirection.transpose()<<std::endl;

	Eigen::AngleAxisd aa(angle, axis);
	Eigen::Vector3d goalDirection = Eigen::Vector3d(14.0 -1.5 + 0.05, 3.1+0.2, 0.0)- standPosition.segment(3,3);
	if(curDirection.dot(goalDirection) <0.0)
	{
		aa = Eigen::AngleAxisd(M_PI, Eigen::Vector3d::UnitY())*aa;
	}
	Eigen::Vector3d newRootOrientation = aa.angle() * aa.axis();

	standPosition.segment(0,3) = newRootOrientation;

	criticalPoint_targetBallPosition = curBallPosition;
	criticalPoint_targetBallVelocity.setZero();

	if((double) rand()/RAND_MAX > 0.5)
		dribbleDefaultVec[3] *= -1;

	for(int i=0;i<RESET_ADAPTING_FRAME;i++)
		mMotionGenerator->setCurrentPose(standPosition, Utils::toStdVec(dribbleDefaultVec));

	for(int i=0;i<RESET_ADAPTING_FRAME;i++)
		mMotionGenerator->generateNextPoseAndContacts(Utils::toStdVec(dribbleDefaultVec));
	// Eigen::VectorXd zeroAction = mActions[0];
	// zeroAction.setZero();
	// zeroAction[4+4] = 1;
	auto nextPoseAndContacts = mMotionGenerator->generateNextPoseAndContacts(Utils::toStdVec(dribbleDefaultVec));
    Eigen::VectorXd nextPosition = std::get<0>(nextPoseAndContacts);
    mCharacters[0]->getSkeleton()->setPositions(nextPosition);
    curBallPosition = std::get<1>(nextPoseAndContacts).segment(4,3);
	for(int i=0;i<3;i++)
	{
		updatePrevBallPositions(curBallPosition);
	}

	Eigen::Vector6d ballPosition = ballSkel->getPositions();
	ballPosition.setZero();
	ballPosition.segment(3,3) = curBallPosition;
	ballSkel->setPositions(ballPosition);

	for(int i=0;i<mNumChars;i++)
	{
		mPrevCOMs[i] = mCharacters[i]->getSkeleton()->getRootBodyNode()->getCOM();
	}
	mPrevBallPosition = ballSkel->getCOM();

	for(int i=0;i<mNumChars;i++)
	{
		mPrevBallPossessions[i] = false;
		mCurBallPossessions[i] =false;
		mDribbled[i] = false;

		mLFootDetached[i] = false;
		mRFootDetached[i] = false;
		// bsm[i]->curState = BasketballState::POSITIONING;
		// bsm[i]->prevAction = 4;


		bsm[i]->curState = BasketballState::DRIBBLING;
		bsm[i]->prevAction = 0;

		mLFootContacting[i] = false;
		mRFootContacting[i] = false;

		mLLastFootPosition[i].setZero();
		mRLastFootPosition[i].setZero();

	}
	// for(int i=0;i<mNumChars;i++)
	// {
	// 	mCharacters[i]->getSkeleton()->setPositions(standPosition);
	// }

	curFrame = 0;
	criticalPointFrame = 0; 
	// std::cout<<"B"<<std::endl;

}

void
Environment::
resetTargetBallPosition()
{
	Eigen::Isometry3d rootT = getRootT(0);

	double xRange = 5.0;
	double yRange = 3.0;
	double zRange = 4.0;

	mTargetBallPosition[0] = (double) rand()/RAND_MAX * xRange*0.5 + xRange*0.5;
	// mTargetBallPosition[1] = (double) rand()/RAND_MAX * yRange*0.5 + yRange*0.5 ;
	mTargetBallPosition[1] = 2.4 ;
	mTargetBallPosition[2] = (double) rand()/RAND_MAX * zRange*0.5 + zRange*0.5;

	mTargetBallPosition = rootT * mTargetBallPosition;

	mTargetBallPosition = Eigen::Vector3d(14.0 -1.5 + 0.05, 3.1+0.2, 0.0);
	// mTargetBallPosition = Eigen::Vector3d(14.0 -1.5 + 0.05, 2.0, 0.0);

}

// Eigen::VectorXd
// Environment::
// normalizeNNState(Eigen::VectorXd state)
// {
// 	Eigen::VectorXd normalizedState = state;

// 	return normalizedState;
// }


// Eigen::VectorXd
// Environment::
// unNormalizeNNState(Eigen::VectorXd normalizedState)
// {
// 	Eigen::VectorXd unNormalizedState = normalizedState;
// 	return unNormalizedState;
// }

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
updatePrevContacts(int index, Eigen::Vector2d handContacts)
{
	prevContact[index] = curContact[index];

   curContact[index] = -1;
    if(handContacts[0]>0.5)
        curContact[index] = 0;
    if(handContacts[1]>0.5)
        curContact[index] = 1;
    
    if(handContacts[0]>0.5 && handContacts[1]>0.5)
        curContact[index] = 2;
    // std::cout<<"updatePrevContacts "<<handContacts.transpose()<<std::endl;
    // std::cout<<"updatePrevContacts "<<curContact[index]<<std::endl;

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

Eigen::Vector3d
Environment::
computeHandBallPosition(int index)
{
	assert(curContact[index] != -1);
	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	// Eigen::Vector3d prevBallPositions[0]
	Eigen::Vector3d relBallPosition = Eigen::Vector3d(0.14, 0.16, 0.0);
	if(curContact[index] == 0)
	{
		Eigen::Isometry3d handTransform = skel->getBodyNode("LeftHand")->getWorldTransform();
		Eigen::Vector3d ballPosition = handTransform * relBallPosition;
/*
		Eigen::Vector3d ballPosition = skel->getBodyNode("LeftFingerBall")->getWorldTransform().translation();

		if((ballPosition - curBallPosition).norm() >0.15)
		{
			Eigen::Vector3d ballPositionR = skel->getBodyNode("RightFingerBall")->getWorldTransform().translation();
			if((ballPosition-curBallPosition).norm() > (ballPositionR -curBallPosition).norm())
			{
				curContact[index] = 1;
				return ballPositionR;
			}
			else
			{
				// std::cout<<"Too far ball to left hand"<<std::endl;
				return ballPosition;
			}
		}
		else*/
			return ballPosition;
	}
	else if(curContact[index] == 1)
	{
		Eigen::Isometry3d handTransform = skel->getBodyNode("RightHand")->getWorldTransform();
		Eigen::Vector3d ballPosition = handTransform * relBallPosition;
		/*Eigen::Vector3d ballPosition = skel->getBodyNode("RightFingerBall")->getWorldTransform().translation();
		if((ballPosition - curBallPosition).norm() >0.15)
		{
			Eigen::Vector3d ballPositionL = skel->getBodyNode("LeftFingerBall")->getWorldTransform().translation();
			if((ballPosition-curBallPosition).norm() > (ballPositionL -curBallPosition).norm())
			{
				curContact[index] = 0;
				return ballPositionL;
			}
			else
			{
				// std::cout<<"Too far ball to right hand"<<std::endl;
				return ballPosition;
			}
		}
		else*/
			return ballPosition;
	}
	if(curContact[index] == 2)
	{
		Eigen::Isometry3d leftHandTransform = skel->getBodyNode("LeftHand")->getWorldTransform();
		Eigen::Isometry3d rightHandTransform = skel->getBodyNode("RightHand")->getWorldTransform();
		Eigen::Vector3d ballPosition = Eigen::Vector3d(0.14, 0.16, 0.0);

		ballPosition = (leftHandTransform*relBallPosition + rightHandTransform*relBallPosition)/2.0;
		/*Eigen::Vector3d ballPosition = (skel->getBodyNode("LeftFingerBall")->getWorldTransform().translation() + 
						skel->getBodyNode("RightFingerBall")->getWorldTransform().translation())/2.0;
		if((ballPosition - curBallPosition).norm() >0.15)
		{
			Eigen::Vector3d ballPositionL = skel->getBodyNode("LeftFingerBall")->getWorldTransform().translation();
			Eigen::Vector3d ballPositionR = skel->getBodyNode("RightFingerBall")->getWorldTransform().translation();
			if((ballPositionL-curBallPosition).norm() > (ballPositionR -curBallPosition).norm())
			{
				curContact[index] = 1;
				// std::cout<<"Too far ball to two hand"<<std::endl;
				return ballPositionR;
			}
			else
			{
				curContact[index] = 0;
				// std::cout<<"Too far ball to two hand"<<std::endl;
				return ballPositionL;
			}
		}
		else*/
			return ballPosition;

	}
	// return ballPosition;
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
	int maxIter = 30;
	while(abs(h + v*t1 - g/2.0*pow(t1,2))>1E-3)
	{
		if(maxIter < 0)
			break;
		maxIter--;
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

void 
Environment::
computeCriticalActionTimes()
{
	// std::cout<<"mActions[index][4+8+6] "<<mActions[0][4+8+6]<<std::endl;
	for(int index=0;index<mNumChars;index++)
	{
	    // int curActionType = 0;
	    // int maxIndex = 0;
	    // double maxValue = -100;
	    // for(int i=4;i<12;i++)
	    // {
	    //     if(mActions[index][i] > maxValue)
	    //     {
	    //         maxValue = mActions[index][i];
	    //         maxIndex = i;
	    //     }
	    // }
	    // curActionType = maxIndex-4;

	    // int prevActionType = 0;
	    // maxIndex = 0;
	    // maxValue = -100;
	    // for(int i=4;i<12;i++)
	    // {
	    //     if(mPrevActions[index][i] > maxValue)
	    //     {
	    //         maxValue = mActions[index][i];
	    //         maxIndex = i;
	    //     }
	    // }
	    // prevActionType = maxIndex-4;

		double interp = 0.0;
    	Eigen::Isometry3d rootT = getRootT(index);
	    if(mCurActionTypes[index] == mPrevActionTypes[index])
	    {
	    	if(isCriticalAction(mCurActionTypes[index]))
	    	{
	    		mCurCriticalActionTimes[index]--;
	    		double curActionGlobalBallPosition = mActions[index][4+NUM_ACTION_TYPE+3]/100.0;
	    		Eigen::Vector3d curActionGlobalBallVelocity = rootT.linear() * ( mActions[index].segment(4+NUM_ACTION_TYPE,3)/100.0);

	    		mActionGlobalBallPosition[index] = (1.0-interp) * mActionGlobalBallPosition[index] + interp * curActionGlobalBallPosition;
	    		mActionGlobalBallVelocity[index] = (1.0-interp) * mActionGlobalBallVelocity[index] + interp * curActionGlobalBallVelocity;
	    		// mActions[index].segment(4+6,3) = rootT.inverse() *mActionGlobalBallPosition[index];
	    		mActions[index].segment(4+NUM_ACTION_TYPE,3) = rootT.linear().inverse() * mActionGlobalBallVelocity[index];
	    		mActions[index].segment(4+NUM_ACTION_TYPE,3) *= 100.0;
	    		// mActions[index][4+NUM_ACTION_TYPE+,3] *= mActionGlobalBallPosition[index];

	    		// if(mActions[index].segment(4+6,3).norm()>2000)
	    		// {
	    		// 	std::cout<<mActions[index].segment(4+6,3).transpose()<<std::endl;
	    		// 	exit(0);
	    		// }
	    		mActions[index][4+NUM_ACTION_TYPE+3] = mActionGlobalBallPosition[index]*100.0;
	    	}
	    }
	    else
	    {
	    	mCurCriticalActionTimes[index] = (int) (mActions[index][4+NUM_ACTION_TYPE+4]+0.5);
	    	if(mCurCriticalActionTimes[index] > 30)
	    		mCurCriticalActionTimes[index] = 30;
	    	if(mCurCriticalActionTimes[index] < 20)
	    		mCurCriticalActionTimes[index] = 20;
	    	// mCurCriticalActionTimes[index] = 30;
    		if(mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3)
    		{
    			mActionGlobalBallPosition[index] = mActions[index][4+NUM_ACTION_TYPE+3]/100.0;
    			mActionGlobalBallVelocity[index] = rootT.linear() * (mActions[index].segment(4+NUM_ACTION_TYPE,3)/100.0);
    		}
	    }
	    if(!isCriticalAction(mCurActionTypes[index]))
	    	mCurCriticalActionTimes[index] = 0;
		mActions[index][4+NUM_ACTION_TYPE+4] = mCurCriticalActionTimes[index];
	}
	// std::cout<<"mCurCriticalActionTimes[index] "<<mCurCriticalActionTimes[0]<<std::endl;

}


EnvironmentPackage::EnvironmentPackage()
{
}

void
EnvironmentPackage::
saveEnvironment(Environment* env)
{
	this->mActions = env-> mActions;
	this->mPrevBallPossessions = env-> mPrevBallPossessions;
	this->mCurBallPossessions = env-> mCurBallPossessions;
	this->prevBallPositions = env-> prevBallPositions;
	this->curBallPosition = env-> curBallPosition;
	this->criticalPoint_targetBallPosition = env-> criticalPoint_targetBallPosition;
	this->criticalPoint_targetBallVelocity = env-> criticalPoint_targetBallVelocity;
	this->prevContact = env-> prevContact;
	this->curContact = env-> curContact;
	this->criticalPointFrame = env-> criticalPointFrame;
	this->curFrame = env-> curFrame;
	this->mTargetBallPosition = env-> mTargetBallPosition;
	this->mPrevActions = env-> mPrevActions;
	this->mCurActionTypes = env-> mCurActionTypes;
	this->mPrevCOMs = env-> mPrevCOMs;
	this->mPrevBallPosition = env-> mPrevBallPosition;
	this->mCurCriticalActionTimes = env-> mCurCriticalActionTimes;
	this->mPrevPlayer = env-> mPrevPlayer;
	this->mDribbled = env-> mDribbled;
	this->mLFootDetached = env->mLFootDetached;
	this->mRFootDetached = env->mRFootDetached;
	this->mObstacles = env->mObstacles;
	this->mCurHeadingAngle = env->mCurHeadingAngle;

	this->mLFootContacting = env->mLFootContacting;
	this->mRFootContacting = env->mRFootContacting;
	this->mLLastFootPosition = env->mLLastFootPosition;
	this->mRLastFootPosition = env->mRLastFootPosition;

	env->mMotionGenerator->motionGenerators[0]->saveHiddenState();
}

void
EnvironmentPackage::
copyEnvironmentPackage(EnvironmentPackage* envPack)
{
	// this->skelPosition = env-> skelPosition;
	this->mActions = envPack-> mActions;
	this->mPrevBallPossessions = envPack-> mPrevBallPossessions;
	this->mCurBallPossessions = envPack-> mCurBallPossessions;
	this->prevBallPositions = envPack-> prevBallPositions;
	this->curBallPosition = envPack-> curBallPosition;
	this->criticalPoint_targetBallPosition = envPack-> criticalPoint_targetBallPosition;
	this->criticalPoint_targetBallVelocity = envPack-> criticalPoint_targetBallVelocity;
	this->prevContact = envPack-> prevContact;
	this->curContact = envPack-> curContact;
	this->criticalPointFrame = envPack-> criticalPointFrame;
	this->curFrame = envPack-> curFrame;
	this->mTargetBallPosition = envPack-> mTargetBallPosition;
	this->mPrevActions = envPack-> mPrevActions;
	this->mCurActionTypes = envPack-> mCurActionTypes;
	this->mPrevCOMs = envPack-> mPrevCOMs;
	this->mPrevBallPosition = envPack-> mPrevBallPosition;
	this->mCurCriticalActionTimes = envPack-> mCurCriticalActionTimes;
	this->mPrevPlayer = envPack-> mPrevPlayer;
	this->mDribbled = envPack-> mDribbled;
	this->mLFootDetached = envPack->mLFootDetached;
	this->mRFootDetached = envPack->mRFootDetached;
	this->mObstacles = envPack->mObstacles;
	this->mCurHeadingAngle = envPack->mCurHeadingAngle;


	this->mLFootContacting = envPack->mLFootContacting;
	this->mRFootContacting = envPack->mRFootContacting;
	this->mLLastFootPosition = envPack->mLLastFootPosition;
	this->mRLastFootPosition = envPack->mRLastFootPosition;

}

void
EnvironmentPackage::
restoreEnvironment(Environment* env)
{
	env->mActions = this->mActions;
	env->mPrevBallPossessions = this->mPrevBallPossessions;
	env->mCurBallPossessions = this->mCurBallPossessions;
	env->prevBallPositions = this->prevBallPositions;
	env->curBallPosition = this->curBallPosition;
	env->criticalPoint_targetBallPosition = this->criticalPoint_targetBallPosition;
	env->criticalPoint_targetBallVelocity = this->criticalPoint_targetBallVelocity;
	env->prevContact = this->prevContact;
	env->curContact = this->curContact;
	env->criticalPointFrame = this->criticalPointFrame;
	env->curFrame = this->curFrame;
	env->mTargetBallPosition = this->mTargetBallPosition;
	env->mPrevActions = this->mPrevActions;
	env->mCurActionTypes = this->mCurActionTypes;
	env->mPrevCOMs = this->mPrevCOMs;
	env->mPrevBallPosition = this->mPrevBallPosition;
	env->mCurCriticalActionTimes = this->mCurCriticalActionTimes;
	env->mPrevPlayer = this->mPrevPlayer;
	env->mDribbled = this->mDribbled;
	env->mLFootDetached = this->mLFootDetached;
	env->mRFootDetached = this->mRFootDetached;
	env->mObstacles = this->mObstacles;
	env->mCurHeadingAngle = this->mCurHeadingAngle;
	
	env->mLFootContacting = this->mLFootContacting;
	env->mRFootContacting = this->mRFootContacting;
	env->mLLastFootPosition = this->mLLastFootPosition;
	env->mRLastFootPosition = this->mRLastFootPosition;

	env->mMotionGenerator->motionGenerators[0]->restoreHiddenState();

}

void 
Environment::
updateHeightMap(int index)
{
	Eigen::Vector3d charPos = mCharacters[index]->getSkeleton()->getRootBodyNode()->getCOM();
	for(int i=0;i<mNumGrids;i++)
	{
		for(int j=0;j<mNumGrids;j++)
		{
			mHeightMaps[index][i][j] = 0.0;
		}
	}

	double gridSize = mMapRange/(mNumGrids-1);
	Eigen::Vector3d centerPosition(gridSize*(mNumGrids-1)/2.0, 0.0, gridSize*(mNumGrids-1)/2.0);
	Eigen::Vector3d charPosition(-gridSize*(mNumGrids-1)/4.0, 0.0, 0.0);
	for(int i=0;i<mNumGrids;i++)
	{
		for(int j=0;j<mNumGrids;j++)
		{
			Eigen::Vector3d gridPosition(gridSize*i, 0.0, gridSize*j);
			gridPosition -= centerPosition;
			gridPosition -= charPosition;

			gridPosition = Eigen::AngleAxisd(-mCurHeadingAngle[index], Eigen::Vector3d(0.0, 1.0, 0.0))*gridPosition;
			
			gridPosition += charPos;
			gridPosition[1] = 0.0;

			for(int idx=0;idx<mObstacles.size();idx++)
			{
				if((gridPosition-mObstacles[idx]).norm()<0.4)
				{
					mHeightMaps[index][i][j] = 1.0;
				}
			}

		}
	}
}

std::vector<Eigen::Vector3d>
Environment::
getHeightMapGrids(int index)
{
	Eigen::Vector3d charPos = mCharacters[index]->getSkeleton()->getRootBodyNode()->getCOM();
	std::vector<Eigen::Vector3d> grids;

	double gridSize = mMapRange/(mNumGrids-1);
	Eigen::Vector3d centerPosition(gridSize*(mNumGrids-1)/2.0, 0.0, gridSize*(mNumGrids-1)/2.0);
	Eigen::Vector3d charPosition(-gridSize*(mNumGrids-1)/4.0, 0.0, 0.0);
	for(int i=0;i<mNumGrids;i++)
	{
		for(int j=0;j<mNumGrids;j++)
		{
			Eigen::Vector3d gridPosition(gridSize*i, 0.0, gridSize*j);
			gridPosition -= centerPosition;
			gridPosition -= charPosition;

			gridPosition = Eigen::AngleAxisd(-mCurHeadingAngle[index], Eigen::Vector3d(0.0, 1.0, 0.0))*gridPosition;
			
			gridPosition += charPos;

			gridPosition[1] = 0.05;

			grids.push_back(gridPosition);
		}
	}
	return grids;
}

void
Environment::
goBackEnvironment()
{
	mPrevEnvSituations[1]->restoreEnvironment(this);
	// saveEnvironment();
	for(int i=0;i<2;i++)
	{
		saveEnvironment();
	}
	curFrame++;

}

void
Environment::
saveEnvironment()
{
	if(curFrame%20 == 0)
	{
		mPrevEnvSituations[1]->copyEnvironmentPackage(mPrevEnvSituations[0]);
		mPrevEnvSituations[0]->saveEnvironment(this);
	}
}

BStateMachine::BStateMachine()
{
	// curState = BasketballState::POSITIONING;
	// prevAction = 4;


	curState = BasketballState::DRIBBLING;
	prevAction = 0;
}

int
BStateMachine::
transition(int action, bool transitState)
{
	if(curState == BasketballState::POSITIONING)
	{
		if(action == 4 || action == 5)
		{
			prevAction = action;
			return action;
		}
		else if(action == 2)
		{
			if(transitState)
				curState = BasketballState::BALL_CATCH_1;
			prevAction = action;
			return 2;
		}
		else
		{
			return prevAction;
		}
	}
	else if(curState == BasketballState::BALL_CATCH_1)
	{
		if(action == 6 || action == 7)
		{
			prevAction = action;
			return action;
		}
		else if(action == 0)
		{
			if(transitState)
				curState = BasketballState::DRIBBLING;
			prevAction = action;
			return action;
		}
		else if(action == 1 || action == 3)
		{
			if(transitState)
				curState = BasketballState::POSITIONING;
			prevAction = action;
			return action;
		}
		else
		{
			return prevAction;
		}
	}
	else if(curState == BasketballState::DRIBBLING)
	{
		if(action == 0)
		{
			prevAction = action;
			return action;
		}
		else if(action == 1 || action == 3)
		{
			if(transitState)
				curState = BasketballState::POSITIONING;
			prevAction = action;
			return action;
		}
		else if(action == 6 || action == 7)
		{
			if(transitState)
				curState = BasketballState::BALL_CATCH_2;
			prevAction = action;
			return action;
		}
		else
		{
			return prevAction;
		}
	}
	else if(curState == BasketballState::BALL_CATCH_2)
	{
		if(action == 6 || action == 7)
		{
			if(transitState)
				curState = BasketballState::BALL_CATCH_2;
			prevAction = action;
			return action;
		}
		else if(action == 1 || action == 3)
		{
			if(transitState)
				curState = BasketballState::POSITIONING;
			prevAction = action;
			return action;
		}
		else
		{
			return prevAction;
		}
	}
	else
	{
		std::cout<<"Wrong state in basket ball state!"<<std::endl;
		exit(0);
	}
}

std::vector<int>
BStateMachine::
getAvailableActions()
{
	if(curState == BasketballState::POSITIONING)
	{
		return std::vector<int>{2, 4, 5};
	}
	else if(curState == BasketballState::BALL_CATCH_1)
	{
		return std::vector<int>{0, 1, 3};//, 6, 7};
	}
	else if(curState == BasketballState::DRIBBLING)
	{
		return std::vector<int>{0, 1, 3};//, 6, 7};
	}
	else if(curState == BasketballState::BALL_CATCH_2)
	{
		return std::vector<int>{1, 3};//, 6, 7};

	}
	else
	{
		std::cout<<"Wrong state in basket ball state!"<<std::endl;
		exit(0);
	}
}


bool
BStateMachine::
isAvailable(int action)
{
	std::vector<int> aa = getAvailableActions();
	if(std::find(aa.begin(), aa.end(), action) != aa.end())
		return true;
	else
		return false;
}

int
BStateMachine::
getStateWithInt()
{
	if(curState == BasketballState::POSITIONING)
	{
		return 0;
	}
	else if(curState == BasketballState::BALL_CATCH_1)
	{
		return 1;
	}
	else if(curState == BasketballState::DRIBBLING)
	{
		return 2;
	}
	else if(curState == BasketballState::BALL_CATCH_2)
	{
		return 3;
	}
	else
	{
		std::cout<<"Wrong state in BStateMachine"<<std::endl;
		exit(0);
	}
}


void 
Environment::genObstacleNearCharacter()
{
	double dRange = 3.0;

	double distance = (double) rand()/RAND_MAX * dRange + 2.0;
	double angle = (double) rand()/RAND_MAX * M_PI/2.0 - M_PI/4.0;



	// std::cout<<angle<<std::endl;

	Eigen::Vector3d obstaclePosition = mCharacters[0]->getSkeleton()->getRootBodyNode()->getCOM();
	obstaclePosition += distance * Eigen::Vector3d(cos(mCurHeadingAngle[0] + angle), 0.0, sin(mCurHeadingAngle[0] + angle));
	obstaclePosition[1] = 0.0;
	mObstacles.push_back(obstaclePosition);
	updateHeightMap(0);
}

void
Environment::removeOldestObstacle()
{
	if(mObstacles.size()>=1.0)
		mObstacles.erase(mObstacles.begin());
	// updateHeightMap(0);
}

void
Environment::genObstaclesToTargetDir(int numObstacles)
{
	Eigen::Vector3d charPos = mCharacters[0]->getSkeleton()->getRootBodyNode()->getCOM();
	Eigen::Vector3d charProjectedPos = charPos;
	charProjectedPos[1] =0.0;
	for(int i=0;i<numObstacles;i++)
	{
		Eigen::Vector3d toTargetVector = mTargetBallPosition - mCharacters[0]->getSkeleton()->getRootBodyNode()->getCOM();
		toTargetVector[1] = 0.0;

		double toTargetDistance = toTargetVector.norm();
		toTargetVector.normalize();

		double distance = (double) rand()/RAND_MAX * toTargetDistance*1.4 - toTargetDistance*0.2;
		double rightDistance = (double) rand()/RAND_MAX * 4.0 - 2.0;

		Eigen::Vector3d toTargetRightVector = Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitY())*toTargetVector;

		while((distance * toTargetVector + rightDistance * toTargetRightVector).norm()<=1.5
			|| (charPos + distance * toTargetVector + rightDistance * toTargetRightVector - mTargetBallPosition).norm()<=1.5)
		{
			distance = (double) rand()/RAND_MAX * toTargetDistance*1.4 - toTargetDistance*0.2;
			rightDistance = (double) rand()/RAND_MAX * 4.0 - 2.0;
		}

		if(i == 0)
		{
			distance = 1.3;
			rightDistance = 0.8;
		}
		else
		{
			distance = 3.2;
			rightDistance = -0.8;
		}

		Eigen::Vector3d obstaclePosition =charPos + distance * toTargetVector + rightDistance * toTargetRightVector;
		obstaclePosition[1] = 0.0;
		mObstacles.push_back(obstaclePosition);
	}
}


void
Environment::genObstacleNearGoalpost()
{
	double dRange = 1.0;

	double distance = 2.0;
	double angle = (double) rand()/RAND_MAX * M_PI/1.0;// - M_PI/4.0;



	// std::cout<<angle<<std::endl;

	Eigen::Vector3d obstaclePosition = mTargetBallPosition;
	obstaclePosition += distance * Eigen::Vector3d(cos(M_PI/2.0 + angle), 0.0, sin(M_PI/2.0 + angle));
	obstaclePosition[1] = 0.0;
	mObstacles.push_back(obstaclePosition);
	// updateHeightMap(0);
}

Eigen::Isometry3d
Environment::
getRootT(int index)
{
	SkeletonPtr skel = mCharacters[index]->getSkeleton();

	Eigen::Isometry3d rootT = skel->getRootBodyNode()->getWorldTransform();

	Eigen::Vector3d front = rootT.linear()*Eigen::Vector3d::UnitY();

	// front = Eigen::AngleAxisd(skel->getPositions().segment(0,3).norm(), skel->getPositions().segment(0,3).normalized())*Eigen::Vector3d::UnitY();
	front[1] = 0.0;
	front.normalize();


	Eigen::Isometry3d rootIsometry;

	rootIsometry.linear() << front[0], 0.0, -front[2],
							0.0, 1.0, 0.0,
							front[2], 0.0, front[0];

	rootIsometry.translation() = rootT.translation();
	rootIsometry.translation()[1] = 0.0;
	return rootIsometry;
}


void
Environment::
applyAction(int index)
{
	// for(int i=4;i<12;i++)
	// {
	// 	mActions[index][i] = 0;
	// }
	// mActions[index][4] = 1;

	// mActions[index][18] = 0;

	// mActions[index].segment(4+8,6).setZero();

	// std::cout<<mActions[index].segment(0,4).transpose()<<std::endl;
	// std::cout<<mActions[index].segment(4,6).transpose()<<std::endl;
	// std::cout<<mActions[index].segment(10,6).transpose()<<std::endl;
	// std::cout<<mActions[index].segment(16,2).transpose()<<std::endl;
	// std::cout<<endl;



	// Eigen::VectorXd mgAction = mActions[index].segment(0,15);

	auto nextPoseAndContacts = mMotionGenerator->generateNextPoseAndContacts(Utils::toStdVec(getMGAction(index)));

	mPrevBallPosition = ballSkel->getCOM();
	for(int i=0;i<mNumChars;i++)
	{
		mPrevCOMs[i] = mCharacters[i]->getSkeleton()->getRootBodyNode()->getCOM();
		mPrevBallPossessions[i] = mCurBallPossessions[i];
	    // mPrevLHandTranform[index] = mCharacters[i]->getSkeleton()->getBodyNode("LeftHand")->getWorldTransform();
	    // mPrevRHandTranform[index] = mCharacters[i]->getSkeleton()->getBodyNode("RightHand")->getWorldTransform();

	}


    Eigen::VectorXd nextPosition = std::get<0>(nextPoseAndContacts);

    mCharacters[index]->mSkeleton->setPositions(nextPosition);
    mCharacters[index]->mSkeleton->setVelocities(mCharacters[0]->mSkeleton->getVelocities().setZero());


    Eigen::Vector4d nextContacts = std::get<1>(nextPoseAndContacts).segment(0,4);
    // mCurActionTypes[index] = std::get<2>(nextPoseAndContacts);
	mCurBallPossessions[index] = std::get<2>(nextPoseAndContacts);
	if(mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3)
	{

		// if(mCurCriticalActionTimes[index] < -10)
		// {
		// 	mCurBallPossessions[index] = false;
		// }
	}
	if(mCurActionTypes[index] == 2)
	{
		// if(mCurCriticalActionTimes[index] > 0)
		// {
		// 	mCurBallPossessions[index] = false;
		// }
		// else
		// 	mCurBallPossessions[index] = true;
	}
	if(mCurActionTypes[index] == 4 || mCurActionTypes[index] == 5)
	{
		mCurBallPossessions[index] = false;
	}
	if(mCurActionTypes[index] == 0 || mCurActionTypes[index] == 6 || mCurActionTypes[index] == 7)
	{
		mCurBallPossessions[index] = true;
	}

    // std::cout<<mActions[index].transpose()<<std::endl;

	// mCurActionTypes[index] = mActions[index];


    // int curActionType = 0;
    // int maxIndex = 0;
    // double maxValue = -100;
    // for(int i=4;i<12;i++)
    // {
    //     if(mActions[index][i] > maxValue)
    //     {
    //         maxValue = mActions[index][i];
    //         maxIndex = i;
    //     }
    // }

    // curActionType = maxIndex-4;


    
 //    if(!mCurBallPossessions[index])
	// {
	// 	// std::cout<<"--"<<std::endl;
	// 	curBallPosition = computeBallPosition();
	// }



    //belows are ball control

	// if(mCurCriticalActionTimes[index]<-10)
	// {
	// 	if((mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3))
	// 	{

	// 	}
	// }


    //Update hand Contacts;
    if(mChangeContactIsActive[index])
    {
    	std::cout<<"mChangeContactIsActive"<<std::endl;
    	updatePrevContacts(index, mActions[index].segment(4+NUM_ACTION_TYPE+5,2));
    	if(curContact[index] == -1)
    	{
    		mChangeContactIsActive[index] = false;
    		mCurBallPossessions[index] = false;
    		// std::cout<<"here"<<std::endl;
    	}
    	else
    	{
    		mCurBallPossessions[index] = true;
    	}
    	// std::cout<<curContact[index]<<std::endl;
    	// std::cout<<std::endl;

    }
    else
    {
    	if((mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3) 
    		&& mCurCriticalActionTimes[index]<10
    		&& !mChangeContactIsActive[index])
    	{
    		updatePrevContacts(index, Eigen::Vector2d::Zero());
    		mCurBallPossessions[index] = false;
    	}
    	else
   			updatePrevContacts(index, std::get<1>(nextPoseAndContacts).segment(2,2));

    }

    if((mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3) 
    		&& mCurCriticalActionTimes[index]<-10)
    {
    	curContact[index] = -1;
    }

    // std::cout<<curContact[index]<<std::endl;
    // std::cout<<mCurBallPossessions[index]<<std::endl;

   //  if(curContact[index] == -1)
   //  {
   //  	if(mCurActionTypes[index]==6 || mCurActionTypes[index] == 7)
   //  	{
   //  		curContact[index] = prevContact[index];
   //  		if(prevContact[index] == -1)
   //  			mIsTerminalState = true;
   //  	}
   //  	if(mCurActionTypes[index]==1 || mCurActionTypes[index] == 3)
   //  	{
			// // if(mCurCriticalActionTimes[index]<10 && mCurCriticalActionTimes[index]>0)
   // //  		{
   // //  			curContact[index] = prevContact[index];
   // //  			if(prevContact[index] == -1)
   // //  				mIsTerminalState = true;
   // //  		}		
			// // else
			// // 	curContact[index] = -1;

   //  	}

   //  	// if(mCurActionTypes[index] == 1 ||mCurActionTypes[index] == 3 && mCurCriticalActionTimes[index]>0)
   //  	// {
   //  	// 	curContact[index] = prevContact[index];
   //  	// }
   //  	// else if(mCurActionTypes[index] == 2 && mCurCriticalActionTimes[index]<=0)
   //  	// {
   //  	// 	curContact[index] = prevContact[index];
   //  	// }
   //  }


 //   	if(mCurActionTypes[index] == 2)
	// {
	// }
    // if(bsm[index]->curState = BasketballState::)
    // std::cout<<curContact[index]<<std::endl;


    if(mCurActionTypes[index] == 4 || mCurActionTypes[index] == 5)
    {
    	updatePrevBallPositions(computeBallPosition());
    }
    else if(curContact[index] >= 0)
    {
    // 	if(mCurActionTypes[index] ==1 || mCurActionTypes[index] == 3)
    // 	{
    // 		if(mCurCriticalActionTimes[index] == 1)
    // 		{
    // 			Eigen::Isometry3d curHandTransform;
    // 			if(curContact[index] == 0)
    // 			{
    // 				curHandTransform = mCharacters[index]->getSkeleton()->getBodyNode("LeftHand")->getWorldTransform();
    // 			}
    // 			else if(curContact[index] == 1)
    // 			{
    // 				curHandTransform = mCharacters[index]->getSkeleton()->getBodyNode("RightHand")->getWorldTransform();
    // 			}
    // 			else
    // 			{
    // 				updatePrevBallPositions(computeHandBallPosition(index));
    // 			}

				// Eigen::Vector3d handProjectedBallPosition = curHandTransform.inverse() * prevBallPositions[0];
				// handProjectedBallPosition[1] = 0.0;
				// Eigen::Vector3d handCenter = Eigen::Vector3d(0.05,0.0,0.0);
				// Eigen::Vector3d ballOffset = (handProjectedBallPosition-handCenter).normalized()* 0.15;
				// ballOffset[1] = 0.23;
				// ballOffset += handCenter;
				// updatePrevBallPositions(curHandTransform*ballOffset);
    // 		}
    // 		else
    // 		{
    // 			updatePrevBallPositions(computeHandBallPosition(index));
    // 		}

    // 	}
    // 	else
    // 	{
    		updatePrevBallPositions(computeHandBallPosition(index));
    	// }
    }
    else if(mCurActionTypes[index] == 0)
    {
		updatePrevBallPositions(std::get<1>(nextPoseAndContacts).segment(4,3));
    }
    else if(mCurBallPossessions[index])
    {
    	updatePrevBallPositions(std::get<1>(nextPoseAndContacts).segment(4,3));
    }
    else
    {
    	updatePrevBallPositions(computeBallPosition());
    }


    Eigen::Vector6d ballPosition;
    ballPosition.setZero();

    ballPosition.segment(3,3) = curBallPosition;

    ballSkel->setPositions(ballPosition);

    Eigen::Vector6d zeroVelocity;
    zeroVelocity.setZero();
    ballSkel->setVelocities(zeroVelocity);

  //   if(mCurBallPossessions[index])
		// mPrevPlayer = index;

    mPrevActionTypes[index] = mCurActionTypes[index];





    // check rule violation

    //double dribble
 //    if(mPrevPlayer== index && !mPrevBallPossessions[index] && mCurBallPossessions[index])
 //    {
 //    	// std::cout<<"Foul : double dribble"<<std::endl;
	// 	// mIsTerminalState = true;
	// 	mIsFoulState = true;
 //    }

	// // pass recieve
	// if(mCurActionTypes[index] == 2 && mCurBallPossessions[index])
	// 	mPrevPlayer = index;

	// // predict ball possession before pass recieve
	// if(mPrevPlayer != index && mCurBallPossessions[index])
	// {
	// 	// std::cout<<"Getting ball before pass receive"<<std::endl;
	// 	// mIsTerminalState = true;
	// 	mIsFoulState = true;
	// }


    //** get foot contacting
	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	if(std::get<1>(nextPoseAndContacts)[0] > 0.5)
	{
		if(!mLFootContacting[index])
			mLLastFootPosition[index] = skel->getBodyNode("LeftToe")->getWorldTransform().translation();
		mLFootContacting[index] = true;
	}
	else
	{
		mLFootContacting[index] = false;
	}

	if(std::get<1>(nextPoseAndContacts)[1] > 0.5)
	{
		if(!mRFootContacting[index])
			mRLastFootPosition[index] = skel->getBodyNode("RightToe")->getWorldTransform().translation();
		mRFootContacting[index] = true;
	}
	else
	{
		mRFootContacting[index] = false;
	}


	if(mLFootContacting[index])
	{
		Eigen::Vector3d curLFootPosition = skel->getBodyNode("LeftToe")->getWorldTransform().translation();
		Eigen::Vector3d footDiff = curLFootPosition - mLLastFootPosition[index];
		footDiff[1] = 0.0;
		// std::cout<<(curLFootPosition - mLLastFootPosition[index]).norm()<<std::endl;
		if(footDiff.norm()>0.35)
		{
			// std::cout<<"Left Foot Sliding"<<std::endl;
			mIsTerminalState = true;
		}
	}

	if(mRFootContacting[index])
	{
		Eigen::Vector3d curRFootPosition = skel->getBodyNode("RightToe")->getWorldTransform().translation();
		Eigen::Vector3d footDiff = curRFootPosition - mRLastFootPosition[index];
		footDiff[1] = 0.0;
		if(footDiff.norm()>0.35)
		{
			// std::cout<<"Right Foot Sliding"<<std::endl;
			mIsTerminalState = true;
		}
	}


	// if(mCurBallPossessions[index] && curContact[index] == -1)
	// {
	// 	if(mCurActionTypes[index] != 0)
	// 	{
	// 		mIsTerminalState = true;
	// 	}

	// }

	/*if((curBallPosition - mCharacters[index]->getSkeleton()->getRootBodyNode()->getCOM()).norm() < 0.10)
	{
		mIsTerminalState = true;
	}*/


	// if(mCurActionTypes[index] == 0)
	// {
	// 	mDribbled[index] = true;
	// 	mLFootDetached[index] = false;
	// 	mRFootDetached[index] = false;
	// }
	// if(mCurActionTypes[index] == 1 || mCurActionTypes[index] == 3)
	// {
	// 	if(!mCurBallPossessions[index])
	// 	{
	// 		mDribbled[index] = false;
	// 			mPrevPlayer = -1;
	// 	}
	// }





    // if(curActionType == 1 || curActionType == 3)
    // {
    	//Let's check if this is critical point or not

    if(mCurBallPossessions[index])
    {
        this-> criticalPoint_targetBallPosition = this->prevBallPositions[0];
        this-> criticalPoint_targetBallVelocity = (this->prevBallPositions[0] - this->prevBallPositions[2])*15*1.5;
        if(this-> criticalPoint_targetBallVelocity.norm() >8)
        {
        	this->criticalPoint_targetBallVelocity = 8.0 * this->criticalPoint_targetBallVelocity.normalized();
        }
        this->criticalPointFrame = curFrame;
    }

// 	if(prevContact[index] != -1 && curContact[index] == -1)
// 	{
//         this-> criticalPoint_targetBallPosition = this->prevBallPositions[0];
//         this-> criticalPoint_targetBallVelocity = (this->prevBallPositions[0] - this->prevBallPositions[2])*15*1.5;
//         if(this-> criticalPoint_targetBallVelocity.norm() >8)
//         {
//         	this->criticalPoint_targetBallVelocity = 8.0 * this->criticalPoint_targetBallVelocity.normalized();
//         }
//         // std::cout<<"this->prevBallPosition : "<<this->prevBallPosition.transpose()<<std::endl;
//         // std::cout<<"this->pprevBallPosition : "<<this->pprevBallPosition.transpose()<<std::endl;
//         // std::cout<<"this->ppprevBallPosition : "<<this->ppprevBallPosition.transpose()<<std::endl;
        

//         this->criticalPointFrame = curFrame-1;
// 	}

    // }
	// std::cout<<"################# "<<mCurActionTypes[index]<< "###############"<<std::endl;



	// if(mCurActionTypes[index] != 0 && !curContact)



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
