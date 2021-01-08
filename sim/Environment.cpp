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
#include <Eigen/Geometry>
#include "../vsports/common.h"
#include "../extern/ICA/Utils/PathManager.h"
#include "../extern/ICA/CharacterControl/MotionRepresentation.h"
#include "../extern/ICA/Motion/MotionSegment.h"
#include "../extern/ICA/Motion/Pose.h"
#include "../extern/ICA/Motion/RootTrajectory.h"
#include "../extern/ICA/Utils/Functions.h"

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
criticalPointFrame(0), curFrame(0), mIsFoulState(false), gotReward(false), violatedFrames(0),curTrajectoryFrame(0),
randomPointTrajectoryStart(false), resetDuration(10), typeFreq(10), savedFrame(0), foulResetCount(0), curReward(0)
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

	this->nnPath = nn_path;
	mNormalizer = new Normalizer("../extern/ICA/motions/"+nn_path+"/data/xNormal.dat", 
								"../extern/ICA/motions/"+nn_path+"/data/yNormal.dat");
}

void 
Environment::initialize(ICA::dart::MotionGeneratorBatch* mgb, int batchIndex, bool initTutorialTrajectory)
{
	mMgb = mgb;
	mBatchIndex = batchIndex;

    mWorld->setGravity(Eigen::Vector3d(0.0, -9.81, 0.0));



	this->criticalPoint_targetBallPosition = Eigen::Vector3d(0.0, 0.85, 0.0);
	this->criticalPoint_targetBallVelocity = Eigen::Vector3d(0.0, 0.0, 0.0);
	criticalPointFrame = 0;
	curFrame = 0;


	this->resetTargetBallPosition();
	if(initTutorialTrajectory)
		this->genRewardTutorialTrajectory();
	// else

	this->slaveReset();

	mPrevEnvSituation = new EnvironmentPackage(this, mNumChars);
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
copyTutorialTrajectory(Environment* env)
{
	this->mTutorialTrajectories = env->mTutorialTrajectories;
	this->mTutorialBallPositions = env->mTutorialBallPositions;
	this->mTutorialControlVectors = env->mTutorialControlVectors;
}

void
Environment::
setPositionFromBVH(int index, int bvhFrame)
{
	BVHmanager::setPositionFromBVH(mCharacters[index]->getSkeleton(), mBvhParser, bvhFrame);
}


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

	pb2j.setIdentity();
	pb2j.translation() = Eigen::Vector3d(0.095, 0.0, 0.0);
	cb2j.setIdentity();
	cb2j.translation() = Eigen::Vector3d(-fingerLength/2.0, 0.0, 0.0);
	SkelMaker::makeWeldJointBody("LeftFinger", skel, skel->getBodyNode("LeftHand"), SHAPE_TYPE::BOX, Eigen::Vector3d(fingerLength, 0.03, 0.05),
	 pb2j, cb2j);



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


	// in the case of single player
	mCharacters[0]->mSkeleton = bvhSkel;

	mCharacters[0]->prevSkelPositions.resize(mCharacters[0]->mSkeleton->getNumDofs());


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
	// std::cout<<mActions[index].segment(4,NUM_ACTION_TYPE).transpose()<<std::endl;
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
stepAtOnce(std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool> nextPoseAndContacts)
{
	// int sum = 0;
	// for(int i=0;i<mMgb->motionGenerators[0]->mMotionSegments.size();i++)
	// {
	// 	sum += mMgb->motionGenerators[0]->mMotionSegments[i]->mPoses.size();
	// }
	// std::cout<<"mBatchIndex "<<mBatchIndex<<" Total mPoses : "<<sum<<std::endl;

	// if(mBatchIndex<5)
		// std::cout<<mBatchIndex<<": "<<curFrame<<", "<<resetCount<<std::endl;


	saveEnvironment();
	mCharacters[0]->prevSkelPositions = mCharacters[0]->getSkeleton()->getPositions();
	mCharacters[0]->prevKeyJointPositions = mStates[0].segment(mCharacters[0]->getSkeleton()->getNumDofs(),6*3);
	mCharacters[0]->prevRootT = getRootT(0);
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
			// if(mCurCriticalActionTimes[index] >=0)
			// {
			// 	mCurBallPossessions[index] = true;
			// }

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
	if(resetCount < 0)
		resetCount = -1;
	curFrame++;
	curTrajectoryFrame++;

	foulResetCount--;
	if(foulResetCount<=0)
		foulResetCount = 0;

	if(curTrajectoryFrame > mTutorialControlVectors[0].size()-1)
		curTrajectoryFrame = mTutorialControlVectors[0].size()-1;
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

	Eigen::Vector4d rootTransform = ICAPosition.segment(4,4);

	// Motion::Root root = ms->getLastPose()->getRoot();

	// Eigen::Isometry3d baseToRoot = ICA::dart::getBaseToRootMatrix(root);

	// ICAPosition.segment(MotionRepresentation::posOffset,3) = relCurBallPosition;


	SkeletonPtr skel = mCharacters[index]->getSkeleton();
	Eigen::Isometry3d rootT = getRootT(index);

	std::vector<std::string> EEJoints;
	EEJoints.push_back("LeftHand");
	EEJoints.push_back("RightHand");

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

	Eigen::VectorXd skelVelocity(skel->getNumDofs() + 3*EEJoints.size());

	// std::cout<<"###000000"<<std::endl;
	skelVelocity.segment(0, skel->getNumDofs()) = 
	skel->getPositionDifferences(skel->getPositions(), mCharacters[index]->prevSkelPositions);
	// std::cout<<"###111111"<<std::endl;

	if(resetCount<=0)
	{
		skelVelocity.segment(skel->getNumDofs(),3*EEJoints.size()) 
		= skelPosition.segment(skel->getNumDofs(),3*EEJoints.size()) - mCharacters[index]->prevKeyJointPositions;
	}
	else
	{
		skelVelocity.segment(skel->getNumDofs(),3*EEJoints.size()).setZero();
	}


	skelVelocity.segment(3,3) = rootT.linear().inverse() * skelVelocity.segment(3,3);
	// std::cout<<"###222222"<<std::endl;

	// Eigen::VectorXd rootState(3 + 3);
	// rootState.segment(0,3) = skelPosition.segment(3,3);
	// rootState.segment(3,3) = skelVelocity.segment(3,3);

	// rootState.segment(3,3) = rootT.linear().inverse() * rootState.segment(3,3);

	Eigen::VectorXd reducedSkelPosition(6);
	reducedSkelPosition = skelPosition.segment(0,6);

	Eigen::VectorXd reducedSkelVelocity(6);
	reducedSkelVelocity = skelVelocity.segment(0,6);



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
		curBallVelocity = 60.0 * (curBallPosition - prevBallPositions[1]);
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
	Eigen::Vector2d curSMState;
	curSMState.setZero();
	curSMState[bsm[index]->getStateWithInt()] = 1.0;

	Eigen::VectorXd availableActions(2);
	availableActions.setZero();
	std::vector<int> availableActionList = bsm[index]->getAvailableActions();

	if((mCurActionTypes[index] == 3)
		&&mCurCriticalActionTimes[index] >-40)
	{
		availableActions[mCurActionTypes[index]/3] = 1;
	}
	else
	{
		for(int i=0;i<availableActionList.size();i++)
		{
			availableActions[availableActionList[i]] = 1;
		}
	}


	for(int i=0;i<availableActions.size();i++)
	{
		mCharacters[index]->availableActionTypes[i] = availableActions[i];
	}


	std::vector<Eigen::Vector3d> relObstacles(mObstacles.size());
	for(int i=0;i<mObstacles.size();i++)
	{
		relObstacles[i] = rootT.inverse()*mObstacles[i];
		// relObstacles[i][1] = 0;
	}

	// std::cout<<"relObstacles[0].transpose() : "<<relObstacles[0].transpose()<<std::endl;

	// mCharacters[index]->blocked = false;
	// for(int i=0;i<relObstacles.size();i++)
	// {
	// 	Eigen::Vector3d projectedGoalpostPositions = goalpostPositions.segment(0,3);
	// 	projectedGoalpostPositions[1] = 0.0;
	// 	Eigen::Vector3d temp = projectedGoalpostPositions.normalized() * relObstacles[i].dot(projectedGoalpostPositions.normalized());
	// 	// std::cout<<"temp.norm() : "<<temp.norm()<<std::endl;
	// 	// std::cout<<"projectedGoalpostPositions.norm() : "<<projectedGoalpostPositions.norm()<<std::endl;
	// 	if(temp.norm() > projectedGoalpostPositions.norm())
	// 		continue;
	// 	temp[1] = relObstacles[i][1];
	// 	// std::cout<<"Temp : "<<temp.transpose()<<std::endl;
	// 	// std::cout<<"relObstacles : "<<relObstacles[i].transpose()<<std::endl;
	// 	double distance = (relObstacles[i] - temp).norm();
	// 	// std::cout<<distance<<std::endl;
	// 	if(distance < 0.5)
	// 	{
	// 		mCharacters[index]->blocked = true;
	// 		break;
	// 	}
	// }


	mCharacters[index]->blocked = false;
	Eigen::Vector3d targetPlaneNormal = mObstacles[0] - mCharacters[index]->getSkeleton()->getRootBodyNode()->getCOM();
	targetPlaneNormal[1] = 0.0;

	mCharacters[index]->blocked = targetPlaneNormal.norm()<0.75;


	// for(int i=0;i<relObstacles.size();i++)
	// {


	// 	Eigen::Vector3d projectedGoalpostPositions = goalpostPositions.segment(0,3);
	// 	projectedGoalpostPositions[1] = 0.0;
	// 	Eigen::Vector3d temp = projectedGoalpostPositions.normalized() * relObstacles[i].dot(projectedGoalpostPositions.normalized());
	// 	// std::cout<<"temp.norm() : "<<temp.norm()<<std::endl;
	// 	// std::cout<<"projectedGoalpostPositions.norm() : "<<projectedGoalpostPositions.norm()<<std::endl;
	// 	if(temp.norm() > projectedGoalpostPositions.norm())
	// 		continue;
	// 	temp[1] = relObstacles[i][1];
	// 	// std::cout<<"Temp : "<<temp.transpose()<<std::endl;
	// 	// std::cout<<"relObstacles : "<<relObstacles[i].transpose()<<std::endl;
	// 	double distance = (relObstacles[i] - temp).norm();
	// 	// std::cout<<distance<<std::endl;
	// 	if(distance < 0.5)
	// 	{
	// 		mCharacters[index]->blocked = true;
	// 		break;
	// 	}
	// }


	// state.resize(3);

/*
	state.resize(rootTransform.rows() + reducedSkelPosition.rows() + reducedSkelVelocity.rows() + relCurBallPosition.rows() + relObstacles.size()*3
		+ 5 +availableActions.rows() + relTargetPosition.rows() + relBallToTargetPosition.rows() + goalpostPositions.rows() + 1 + 1 + curActionType.rows()
		+ curSMState.rows());

	int curIndex = 0;
	for(int i=0;i<rootTransform.rows();i++)
	{
		state[curIndex] = rootTransform[i];
		curIndex++;
	}
	for(int i=0;i<reducedSkelPosition.rows();i++)
	{
		state[curIndex] = reducedSkelPosition[i];
		curIndex++;
	}
	for(int i=0;i<reducedSkelVelocity.rows();i++)
	{
		state[curIndex] = reducedSkelVelocity[i];
		curIndex++;
	}
	for(int i=0;i<relCurBallPosition.rows();i++)
	{
		state[curIndex] = relCurBallPosition[i];
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
	for(int i=0;i<5;i++)
	{
		state[curIndex] = mCharacters[index]->blocked;
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
	// for(int i=0;i<contacts.rows();i++)
	// {
	// 	state[curIndex] = contacts[i];
	// 	curIndex++;
	// }



	for(int i=0;i<availableActions.rows();i++)
	{
		state[curIndex] = availableActions[i];
		curIndex++;
	}









	mStates[index] = state;
	// cout<<"getState end"<<endl;
	return state;
*/


	// std::cout<<" Cur state is "<<bsm[index]->curState<<std::endl;
	// std::cout<<availableActions.transpose()<<std::endl;

	// std::cout<<"getState :"<<std::endl;
	// std::cout<<"skelPosition.transpose(): "<<skelPosition.transpose()<<std::endl;
	// std::cout<<"relCurBallPosition.transpose(): "<<relCurBallPosition.transpose()<<std::endl;
	// std::cout<<"relTargetPosition.transpose(): "<<relTargetPosition.transpose()<<std::endl;
	// std::cout<<"relBallToTargetPosition.transpose(): "<<relBallToTargetPosition.transpose()<<std::endl;
	// std::cout<<"goalpostPositions.transpose(): "<<goalpostPositions.transpose()<<std::endl;
	// std::cout<<"contacts.transpose(): "<<contacts.transpose()<<std::endl;

	bool simplePosition = true;

	if(simplePosition)
	{
		state.resize(rootTransform.rows() + reducedSkelPosition.rows() + reducedSkelVelocity.rows() + relCurBallPosition.rows() 
			+ relTargetPosition.rows() + relBallToTargetPosition.rows() + goalpostPositions.rows() 
			+ contacts.rows() + 1 + 1 + 3 +curActionType.rows()+curSMState.rows() + relObstacles.size()*3 + 5 +availableActions.rows());
	}
	else
	{
		state.resize(rootTransform.rows() + skelPosition.rows() + skelVelocity.rows() + relCurBallPosition.rows() 
		+ relTargetPosition.rows() + relBallToTargetPosition.rows() + goalpostPositions.rows() 
		+ contacts.rows() + 1 + 1 + 3 +curActionType.rows()+curSMState.rows() + relObstacles.size()*3 + 5 +availableActions.rows());

	}


	int curIndex = 0;
	for(int i=0;i<rootTransform.rows();i++)
	{
		state[curIndex] = rootTransform[i];
		curIndex++;
	}
	if(simplePosition)
	{
		for(int i=0;i<reducedSkelPosition.rows();i++)
		{
			state[curIndex] = reducedSkelPosition[i];
			curIndex++;
		}
		for(int i=0;i<reducedSkelVelocity.rows();i++)
		{
			state[curIndex] = reducedSkelVelocity[i];
			curIndex++;
		}
	}
	else
	{
		for(int i=0;i<skelPosition.rows();i++)
		{
			state[curIndex] = skelPosition[i];
			curIndex++;
		}
		for(int i=0;i<skelVelocity.rows();i++)
		{
			state[curIndex] = skelVelocity[i];
			curIndex++;
		}
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


	for(int i=0;i<relObstacles.size();i++)
	{
		state[curIndex] = relObstacles[i][0];
		curIndex++;
		state[curIndex] = relObstacles[i][1];
		curIndex++;
		state[curIndex] = relObstacles[i][2];
		curIndex++;
	}

	for(int i=0;i<5;i++)
	{
		state[curIndex] = mCharacters[index]->blocked;
		curIndex++;
	}

	for(int i=0;i<availableActions.rows();i++)
	{
		state[curIndex] = availableActions[i];
		curIndex++;
	}

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

	bool isDribble = false;
	bool isDribbleAndShoot = true;

	if(isDribbleAndShoot)
	{
		Eigen::Vector3d targetPlaneNormal = mObstacles[0] - mCharacters[index]->getSkeleton()->getRootBodyNode()->getCOM();
		targetPlaneNormal[1] = 0.0;

		Eigen::Vector3d comTargetDirection = targetPlaneNormal.normalized();

		Eigen::Vector3d curRootVelocity = mCharacters[index]->getSkeleton()->getRootBodyNode()->getCOM() - mPrevCOMs[index];

		// if(mCharacters[index]->inputActionType == 3)
		// {
		// 	mCharacters[index]->inputActionType = 0;
		// 	reward -= 0.01* pow(targetPlaneNormal.norm(),2);;
		// }

		// if(mCurActionTypes[index] == 3)
		// {
		// 	reward = exp(-0.3*pow(targetPlaneNormal.norm(),2));
		// 	mIsTerminalState = true;
		// 	return reward;
		// }
		// else
		// {
		// 	return 0;
		// }

		if(mCharacters[index]->blocked)
		{
			// mIsTerminalState = true;
			// curReward = 1.0;
			// return 1.0;

			// if(mCurActionTypes[index] == 3)
			// {
			// 	mIsTerminalState = true;
			// 	return 1.0;
			// }
			// else
			// 	return 0;


			if(gotReward)
				return 0;

			// if(mPrevActionTypes[index] == 0 && mCurActionTypes[index]==3)
			// 	reward += 0.1;


			if(!mCurBallPossessions[index])
			{
				reward += 1.0;
				if(fastViewTermination)
					mIsTerminalState = true;
				gotReward = true;

				Eigen::Vector3d relTargetPosition = mTargetBallPosition - criticalPoint_targetBallPosition;

				double h = relTargetPosition[1];

				double v = criticalPoint_targetBallVelocity[1];
				// reward += 0.02*v*relTargetPosition.normalized().dot(criticalPoint_targetBallVelocity.normalized());

				//vt + 1/2 gt^2 = h
				// std::cout<<"v*v+2*g*h "<<v*v+2*g*h<<std::endl;
				if(v*v+2*g*h<0)
				{
					curReward = reward;
					return reward;
					// return reward;
				}
				double t = (-v -sqrt(v*v+2*g*h))/g;

				Eigen::Vector3d ballPositionOnThePlane = criticalPoint_targetBallPosition + criticalPoint_targetBallVelocity*t;
				ballPositionOnThePlane[1] = 0.0;

				Eigen::Vector3d targetPositionOnThePlane = mTargetBallPosition;
				targetPositionOnThePlane[1] = 0.0;

				// if(!mCharacters[index]->blocked)
					reward += 2.0*exp(0.3 * -pow((targetPositionOnThePlane - ballPositionOnThePlane).norm(),2));
				// else
				// 	reward = 0.0 * exp(0.3 * -pow((targetPositionOnThePlane - ballPositionOnThePlane).norm(),2));

				curReward = reward;
				return reward;
			}
			else
			{
				return reward;
			}

		}
		else
		{
			curReward = 0;
			if(mCurActionTypes[index] == 3)
			{

				// mIsTerminalState = true;
				// curReward = 0;
				// return 0;

				if(!mCurBallPossessions[index])
				{
				
					// mIsFoulState = true;
					mIsTerminalState = true;
					// return -0.01* pow(targetPlaneNormal.norm(),2);
					curReward = 0;
					return 0;
					// return -0.1;
				}

				// return - 0.1*targetPlaneNormal.norm();
				// return 0.1*exp(-(targetPlaneNormal.norm()));
				// return -0.1;
			}
		}

		return reward;

	}

	if(isDribble)
	{
		// if(mCurActionTypes[index] == 0)
		// 	return 0.01;
		// else
		// {
		// 	mIsTerminalState = true;
		// 	return 0;
		// }

		// action type correction reward
		// std::vector<int> availableActionTypes = mCharacters[index]->availableActionTypes;
		// std::vector<int>::iterator iter;

		// if(std::find(availableActionTypes.begin(), 
		// 			availableActionTypes.end(), 
		// 			mCharacters[index]->inputActionType) != availableActionTypes.end())
		// 	reward += 0.001;

		Eigen::Vector3d ballDisplacement = curBallPosition - prevBallPositions[0];

		// std::cout<<"ballDisplacement norm : "<<ballDisplacement.norm()<<std::endl;

		// if(ballDisplacement.norm() > 0.25)
		// 	reward -= pow((ballDisplacement.norm()-0.1),2);


		if(!mCurBallPossessions[index])
		{
			mIsTerminalState = true;
			return 0;
		}

		if(mCurActionTypes[index] != 0)
		{
			mIsTerminalState = true;
			return 0;
		}

		if(mCharacters[index]->blocked == 0)
			return 0.01;
		else 
			return 0;

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

		if(!mCurBallPossessions[index])
		{
			mIsTerminalState = true;
			if(mCharacters[index]->blocked)
				return 1.0;
			else
				return 0.0;
		}
		else
			return 0;

		// if(mCharacters[index]->blocked && mCurActionTypes[index] == 0)
			// reward += 0.01;
		// Shoot Reward
		if(gotReward)
			return 0;


		if(!mCurBallPossessions[index])
		{
			if(fastViewTermination)
				mIsTerminalState = true;
			gotReward = true;

			if(!mCharacters[index]->blocked)
				return 0;
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

			if(mCharacters[index]->blocked)
				reward += exp(0.3 * -pow((targetPositionOnThePlane - ballPositionOnThePlane).norm(),2));
			else
				reward = 0.0 * exp(0.3 * -pow((targetPositionOnThePlane - ballPositionOnThePlane).norm(),2));

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
		if(resetCount<0)
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
		mActions[index][4+0] = 1.0;
		mIsTerminalState = true;
		return;
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

    mActions[index][4+mCurActionTypes[index]] = 1.0;

    // return;



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



    if(mActions[index].segment(4+NUM_ACTION_TYPE,3).norm()>800.0)
    {
    	mActions[index].segment(4+NUM_ACTION_TYPE,3) *= 800.0/mActions[index].segment(4+NUM_ACTION_TYPE,3).norm();
    }

    if(mActions[index][4+NUM_ACTION_TYPE+3] > 300.0)
    {
    	mActions[index][4+NUM_ACTION_TYPE+3]  = 300.0;
    }

    else if(mActions[index][4+NUM_ACTION_TYPE+3] < 200.0)
    {
    	mActions[index][4+NUM_ACTION_TYPE+3] = 200.0;
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
setActionType(int index, int actionType, bool isNew)
{
	// if(actionType == 1)
		// std::cout<<"Action Type : "<<actionType<<std::endl;
	actionType *= 3;

	int curActionType = actionType;

	if(curFrame%typeFreq == 0)
	{
		// std::cout<<"Saved frame : "<<savedFrame<<std::endl;
		// std::cout<<"curFrame frame : "<<curFrame<<std::endl;
		// std::cout<<"-------"<<std::endl;
		if(!isNew)
		{
			exit(0);
		}
	}

	if(curFrame%typeFreq != 0)
	{
		curActionType = mPrevActionTypes[index];
	}

	if(foulResetCount>0)
	{
		// if(rand()%2==0)
			// curActionType = 0;
	}

	// else
	// {
	// 	std::cout<<"##########"<<curActionType<<std::endl;
	// }

	if(isNew)
		mCharacters[index]->inputActionType = actionType;

	if(actionType != 3)
	 	curActionType = 0;

	if(curFrame <=resetDuration+typeFreq-1)
	{
		curActionType =0;
	}
	// curActionType =0;

	// if(!mCharacters[index]->blocked)
	// 	curActionType = 0;

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
	 	// std::cout<<"1111############# "<<curActionType<<std::endl;
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
		// std::cout<<"2222############# "<<curActionType<<std::endl;
    }
    // std::cout<<"Set Action actiontype : "<<curActionType<<std::endl;
    // bsm[index]->transition(curActionType);
    mCurActionTypes[index] = curActionType;

    // std::cout<<__func__<<" mCurActionTypes[index] :"<<mCurActionTypes[index]<<std::endl;

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
	if(mCurCriticalActionTimes[0] <= -40)
		mIsTerminalState= true;

	// if((mCharacters[0]->getSkeleton()->getCOM()-mTargetBallPosition).norm() > 20.0)
	// 	mIsTerminalState = true;

	Eigen::Isometry3d rootT = getRootT(0);
	double goalPostDistance;

	Eigen::Vector3d projectedCOM = rootT.translation();
	projectedCOM[1] = 0.0;

	Eigen::Vector3d projectedGoalpost = mTargetBallPosition;
	projectedGoalpost[1] =0.0;

	goalPostDistance = (projectedGoalpost-projectedCOM).norm();


	if(abs(rootT.translation()[2])>15.0*0.5*1.1 || 
		rootT.translation()[0]>28.0*0.5*1.1 || 
		rootT.translation()[0] < -4.0)
	{
		mIsTerminalState = true;

	}

	// Eigen::Vector3d targetPlaneNormal = mObstacles[0] - mCharacters[0]->getSkeleton()->getRootBodyNode()->getCOM();
	// targetPlaneNormal[1] = 0.0;

	// if(targetPlaneNormal.norm() > 14.0)
	// {
	// 	mIsTerminalState = true;

	// }


	return mIsTerminalState;
	// return false;
}
bool
Environment::
isTimeOut()
{
	return mTimeElapsed > endTime;
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

	// resetCharacterPositions();
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
	resetCount = resetDuration;
	foulResetCount = 0;

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
	// for(int i=0;i<3;i++)
	// 	genObstacleNearGoalpost();

	genObstacleNearGoalpost(-1);
	// genObstacleNearGoalpost(0);
	// genObstacleNearGoalpost(0.4);
	// genObstacleNearGoalpost(1.2);
	// genObstacleNearGoalpost(1.6);
	// genObstacleNearGoalpost(2.8);
	// genObstacleNearGoalpost(3.2);


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
foulReset()
{

	mIsTerminalState = false;
	mIsFoulState = false;
	goBackEnvironment();
	foulResetCount = typeFreq;
	// slaveReset();
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
	double xRange = 28.0*0.5 -1.5;
	double zRange = 15.0*0.5*0.8;

	// int resetDuration;

	setPositionFromBVH(0, rand()%60+100);
	Eigen::VectorXd standPosition = mCharacters[0]->getSkeleton()->getPositions();
	standPosition[4] = 0.895;

	bool isNan = false;
	for(int i=0;i<standPosition.size();i++)
	{
		if(std::isnan(standPosition[i]))
		{
			isNan = true;
		}
	}
	if(isNan)
		standPosition = mTutorialTrajectories[0][0];

	if(randomPointTrajectoryStart)
	{
		int trajectoryLength = mTutorialTrajectories[0].size();
		int randomPoint = rand()%(trajectoryLength-30-resetDuration)+resetDuration;

		curTrajectoryFrame = randomPoint - resetDuration;

		standPosition = mTutorialTrajectories[0][randomPoint-resetDuration];
		
		slaveResetTargetVector = mTutorialControlVectors[0][randomPoint-resetDuration];
		slaveResetTargetTrajectory.clear();
		slaveResetPositionTrajectory.clear();
		slaveResetBallPositionTrajectory.clear();
		for(int i=0;i<resetDuration;i++)
		{
			slaveResetTargetTrajectory.push_back(mTutorialControlVectors[0][randomPoint-resetDuration+i]);
			slaveResetPositionTrajectory.push_back(mTutorialTrajectories[0][randomPoint-resetDuration+i]);
			slaveResetBallPositionTrajectory.push_back(mTutorialBallPositions[0][randomPoint-resetDuration+i]);	
		}

		// std::cout<<"mTutorialBallPositions[0].size() : "<<mTutorialBallPositions[0].size()<<std::endl;
		// std::cout<<"mTutorialBallPositions[0][randomPoint] : "<<mTutorialBallPositions[0][randomPoint].transpose()<<std::endl;
		slaveResetBallPosition = mTutorialBallPositions[0][randomPoint-resetDuration];	
	}
	else
	{
		if(useHalfCourt)
		{
			double r = (double) rand()/RAND_MAX * zRange * 0.6 + zRange*0.4;
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


		if((double) rand()/RAND_MAX > 0.5)
			dribbleDefaultVec[3] *= -1;

		
		slaveResetTargetVector = dribbleDefaultVec;
		slaveResetBallPosition.setZero();
	}

	mCharacters[0]->prevSkelPositions = standPosition;
	// mCharacters[0]->prevKeyJointPositions = mStates[0].segment(standPosition.size(),6*3);

	curBallPosition = slaveResetBallPosition;


	criticalPoint_targetBallPosition = curBallPosition;
	criticalPoint_targetBallVelocity.setZero();



	mMgb->clear(mBatchIndex);
	// std::cout<<"standPosition.transpose() "<<standPosition.transpose()<<std::endl;

	if(randomPointTrajectoryStart)
		slaveResetStateVector = mMgb->dartPositionToCombinedPosition(standPosition, slaveResetBallPosition, mBatchIndex);
	else
		slaveResetStateVector = mMgb->dartPositionToCombinedPosition(standPosition, mBatchIndex);

	// slaveResetStateVector = mMgb->dartPositionToCombinedPosition(standPosition, mBatchIndex);

	slaveResetPositionVector = standPosition;


	criticalPoint_targetBallPosition = curBallPosition;
	criticalPoint_targetBallVelocity.setZero();



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
	savedFrame = 0;
	criticalPointFrame = 0; 
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

			return ballPosition;
	}
	else if(curContact[index] == 1)
	{
		Eigen::Isometry3d handTransform = skel->getBodyNode("RightHand")->getWorldTransform();
		Eigen::Vector3d ballPosition = handTransform * relBallPosition;

			return ballPosition;
	}
	if(curContact[index] == 2)
	{
		Eigen::Isometry3d leftHandTransform = skel->getBodyNode("LeftHand")->getWorldTransform();
		Eigen::Isometry3d rightHandTransform = skel->getBodyNode("RightHand")->getWorldTransform();
		Eigen::Vector3d ballPosition = Eigen::Vector3d(0.14, 0.16, 0.0);

		ballPosition = (leftHandTransform*relBallPosition + rightHandTransform*relBallPosition)/2.0;
		return ballPosition;

	}
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
	for(int index=0;index<mNumChars;index++)
	{
		double interp = 0.0;
    	Eigen::Isometry3d rootT = getRootT(index);
	    if(mCurActionTypes[index] == mPrevActionTypes[index])
	    {
	    	if(isCriticalAction(mCurActionTypes[index]))
	    	{
	    		if(mCurCriticalActionTimes[index] > 20)
	    		{
	    			interp = 0.5;
	    		}
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
	    	// if(mCurCriticalActionTimes[index] > 30)
	    	// 	mCurCriticalActionTimes[index] = 30;
	    	// if(mCurCriticalActionTimes[index] < 20)
	    	// 	mCurCriticalActionTimes[index] = 20;
	    	mCurCriticalActionTimes[index] = 30;
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

Eigen::Isometry2d 
Environment::
getLocationDisplacement(Motion::MotionSegment* ms, int start, int end)
{
	if(start > end)
	{
		std::cout<<__func__<<" : wrong input. start is bigger then end"<<std::endl;
		return Eigen::Isometry2d::Identity();
	}
	Motion::Pose* startPose = ms->mPoses[start];
	Motion::Root startRoot = startPose->getRoot();

	Eigen::Isometry2d startT;
	startT.setIdentity();

	startT.translation() = startRoot.pos;

	double cos = startRoot.dir[0];
	double sin = startRoot.dir[1];

	startT.linear() << cos, -sin, sin, cos;


	Motion::Pose* endPose = ms->mPoses[end];
	Motion::Root endRoot = endPose->getRoot();
	Eigen::Isometry2d endT;
	endT.setIdentity();

	endT.translation() = endRoot.pos;

	cos = endRoot.dir[0];
	sin = endRoot.dir[1];

	endT.linear() << cos, -sin, sin, cos;


	Eigen::Isometry2d displacementT = startT.inverse()*endT;

	displacementT.translation() /= 100.0;

	return displacementT;
}

Eigen::Isometry2d
Environment::
getCorrectShootingLocationFromControl(Motion::Pose* criticalPose, std::vector<double> control, double random)
{

	Eigen::Isometry2d rootT = criticalPose->getRootT();

	rootT.translation() /= 100.0;

	Eigen::Vector3d criticalBallPosition = criticalPose->ballPosition/100.0;
	Eigen::Vector2d projectedBallPosition;

	projectedBallPosition[0] = criticalBallPosition[0];
	projectedBallPosition[1] = criticalBallPosition[2];

	projectedBallPosition = rootT.linear()*projectedBallPosition;

	Eigen::Isometry2d identityT;
	identityT.setIdentity();


	Eigen::Vector3d criticalBallVelocity;
	criticalBallVelocity[0] = control[4+5+0];
	criticalBallVelocity[1] = control[4+5+1];
	criticalBallVelocity[2] = control[4+5+2];

	criticalBallVelocity /= 100.0;

	Eigen::Vector2d projectedBallVelocity;
	projectedBallVelocity[0] = criticalBallVelocity[0];
	projectedBallVelocity[1] = criticalBallVelocity[2];

	projectedBallVelocity = rootT.linear()*projectedBallVelocity;

	// criticalBallVelocity = identityT.linear()*criticalBallVelocity;

	Eigen::Vector2d projectedGoalpostPosition;
	// Eigen::Vector3d goalpostPosition = mTargetBallPosition;
	projectedGoalpostPosition[0] = mTargetBallPosition[0];
	projectedGoalpostPosition[1] = mTargetBallPosition[2];


	// first, flying time
	double g = 9.81;
	double h = mTargetBallPosition[1] - criticalBallPosition[1];

	double v_y = criticalBallVelocity[1];

	// std::cout<<"criticalBallPosition[1] : "<<criticalBallPosition[1]<<std::endl;
	// std::cout<<"control[4+5+3]/100.0 : "<<control[4+5+3]/100.0<<std::endl;
	// std::cout<<"criticalBallVelocity : "<<criticalBallVelocity.transpose()<<std::endl;
	// std::cout<<v_y<<std::endl;
	// std::cout<<2*g*h<<std::endl;
	// exit(0);
	double t = (v_y+sqrt(v_y*v_y - 2*g*h))/g;

	Eigen::Vector2d relThrowedBallPosition = projectedBallPosition + projectedBallVelocity*t;
	// std::cout<<"relThrowedBallPosition.norm() : "<<relThrowedBallPosition.norm()<<std::endl;

	// second, find position that do not need rotation
	Eigen::Vector2d defaultPosition = projectedGoalpostPosition - relThrowedBallPosition;


	Eigen::Vector2d goalPostToThrowedPosition = -relThrowedBallPosition;


	Eigen::Vector2d goalPostToThrowedPositionDirection = goalPostToThrowedPosition.normalized();
	double theta = atan2(goalPostToThrowedPositionDirection[1], goalPostToThrowedPositionDirection[0]);


	double targetAngle = theta;


	Eigen::Rotation2Dd rot(targetAngle-theta);

	// std::cout<<"theta  : "<<theta<<std::endl;

	Eigen::Matrix2d rotM(rot);

	Eigen::Isometry2d rotT;
	rotT.setIdentity();
	rotT.linear() = rotM;

	std::cout<<"relThrowedBallPosition.transpose() : "<<relThrowedBallPosition.transpose()<<std::endl;
	std::cout<<"goalPostToThrowedPosition.transpose() : "<<goalPostToThrowedPosition.transpose()<<std::endl;
	std::cout<<"projectedGoalpostPosition.transpose() : "<<projectedGoalpostPosition.transpose()<<std::endl;
	// std::cout<<""

	// identityT.translation() = projectedGoalpostPosition +goalPostToThrowedPosition;
	// identityT = rotT * identityT;
	// identityT.translation() -= rotT.linear() * relThrowedBallPosition;



	identityT.translation() = projectedGoalpostPosition + goalPostToThrowedPosition;
	identityT.linear() = rootT.linear();
	identityT.translation() -= projectedGoalpostPosition;

	// std::cout<<"identityT.translation()11 : "<<identityT.translation().transpose()<<std::endl;
	identityT = rotT * identityT;
	// std::cout<<"identityT.translation()22 : "<<identityT.translation().transpose()<<std::endl;
	identityT.translation() += projectedGoalpostPosition;
	// std::cout<<"identityT.translation()33 : "<<identityT.translation().transpose()<<std::endl;

	// identityT.translation() = projectedGoalpostPosition + rotT * goalPostToThrowedPosition;
	// identityT.linear() = rotT.linear();

	identityT.translation() = projectedGoalpostPosition + goalPostToThrowedPosition;
	identityT.linear() = rootT.linear();



	// std::cout<<"identityT.translation().norm() : "<<identityT.translation().norm()<<std::endl;

	// std::cout<<"rootT.linear() * Eigen::Vector2d(1.0,0.0) : "<<rootT.linear() * Eigen::Vector2d(1.0,0.0)<<std::endl;

	// std::cout<<"rootT.translation() : "<<rootT.translation().transpose()<<std::endl;
	std::cout<<"identityT.translation() : "<<identityT.translation().transpose()<<std::endl;



	// std::cout<<"rootT.inverse()*identityT.translation() : "<<identityT.translation().transpose()<<std::endl;
	// std::cout<<identityT.linear()<<std::endl;

	// identityT.linear() = rootT.linear();

	identityT = rootT.inverse()*identityT;
	return identityT;
}


void 
Environment::
genRewardTutorialTrajectory()
{
	//targetShootingAngle E [0.0, 1.0]
	double targetShootingAngle = 0.5;

	std::string sub_dir = "data";
	std::string xDataPath=  PathManager::getFilePath_data(this->nnPath, sub_dir, "xData.dat", 0 );
	std::string xNormalPath=  PathManager::getFilePath_data(this->nnPath, sub_dir, "xNormal.dat");

	std::string yDataPath=  PathManager::getFilePath_data(this->nnPath, sub_dir, "yData.dat", 0 );
	std::string yNormalPath=  PathManager::getFilePath_data(this->nnPath, sub_dir, "yNormal.dat");

	std::cout<<"Uploaded xData, yData:\n"<<xDataPath<<"\n "<<yDataPath<<std::endl;
	std::cout<<"Start generating guiding trajectory"<<std::endl;

	this->xData.push_back(MotionRepresentation::readXData(xNormalPath, xDataPath, sub_dir));
    this->yData.push_back(std::vector<std::vector<double>>());
    Motion::MotionSegment* ms= MotionRepresentation::restoreMotionSegment(yNormalPath, yDataPath, this->yData[0]);
    std::cout<<ms->mPoses.size()<<std::endl;


	std::vector<int> startFrames;
	std::vector<int> endFrames;

	startFrames	= {222, 3329};
	endFrames 	= {379, 3479};
	// std::cout<<"skeleton dofs : "<<mCharacters[0]->getSkeleton()->getNumDofs()<<std::endl;

	for(int i=0;i<startFrames.size();i++)
	{
		int startFrame = startFrames[i];
		int endFrame = endFrames[i];
		// Eigen::Isometry2d displacementT = getLocationDisplacement(ms, startFrame, endFrame);
		Eigen::Isometry2d correctLocationT = getCorrectShootingLocationFromControl(ms->mPoses[endFrame], xData[0][endFrame], 0.5);
		std::vector<Eigen::VectorXd> tutorialTrajectory;
		std::vector<Eigen::VectorXd> tutorialControlVector;
		std::vector<Eigen::Vector3d> tutorialBallPosition;

		for(int j=0;j<endFrame-startFrame+1;j++)
		{
			// Eigen::Isometry2d displacementT = getLocationDisplacement(ms, startFrame+j, endFrame);
			// Eigen::Isometry2d correctDisplacementT = correctLocationT*displacementT.inverse();

			Eigen::Isometry3d correctDisplacementT = Basic::to3d_i(correctLocationT); 
			Eigen::Isometry3d rootT = Basic::to3d_i(ms->mPoses[startFrame+j]->getRootT());
			Eigen::Isometry3d endRootT = Basic::to3d_i(ms->mPoses[endFrame]->getRootT());
			rootT.translation() /= 100.0;
			endRootT.translation() /= 100.0;

			Eigen::Isometry3d movedRootT = rootT;

			Eigen::VectorXd dartPosition = mMgb->ICAPoseToDartPosition(ms->mPoses[startFrame+j]);
			// if(j==endFrame-startFrame)
			// 	std::cout<<"dartPosition : "<<dartPosition.segment(0,6).transpose()<<std::endl;
			Eigen::AngleAxisd aa = Eigen::AngleAxisd(dartPosition.segment(0,3).norm(), dartPosition.segment(0,3).normalized());


			Eigen::Isometry3d orientation;
			orientation.setIdentity();

			orientation.linear() = aa.toRotationMatrix();
			orientation.translation() = dartPosition.segment(3,3);

			// std::cout<<"dart position angle : "<<aa.angle()<<std::endl;
			// std::cout<<"dart position axis : "<<aa.axis().transpose()<<std::endl;

			Eigen::Isometry3d displacementT = correctDisplacementT;
			displacementT.linear()= correctDisplacementT.linear();
			displacementT.translation()= Eigen::Vector3d(0.0, 0.0, 0.0);



			orientation.translation() += endRootT.linear()*correctDisplacementT.translation();

			movedRootT.translation() += endRootT.linear()*correctDisplacementT.translation();


			Eigen::Vector3d projectedGoalpostPosition;
			// Eigen::Vector3d goalpostPosition = mTargetBallPosition;
			projectedGoalpostPosition[0] = mTargetBallPosition[0];
			projectedGoalpostPosition[1] = 0.0;
			projectedGoalpostPosition[2] = mTargetBallPosition[2];

			//angle E [M_PI*1.0/2.0, M_PI*3.0/2.0]
			// M_PI*1.0/2.0 -> WEST
			// M_PI*3.0/2.0 -> EAST

			Eigen::Vector3d originalShootingLocation = endRootT*correctDisplacementT.translation();
			Eigen::Vector2d curDirection(originalShootingLocation[2] - mTargetBallPosition[2],
										originalShootingLocation[0] - mTargetBallPosition[0]);


			curDirection.normalize();
			// std::cout<<"originalShootingLocation.translation() : "<<originalShootingLocation.transpose()<<std::endl;
			// std::cout<<"mTargetBallPosition.translation() : "<<mTargetBallPosition.transpose()<<std::endl;
			// std::cout<<"curDirection "<<curDirection.transpose()<<std::endl;
			double curAngle = atan2(curDirection[1], curDirection[0]);
			// std::cout<<"curAngle : "<<curAngle<<std::endl;

			double angle = M_PI- curAngle + M_PI*(targetShootingAngle);

			orientation.translation() -= projectedGoalpostPosition;
			orientation = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY())*orientation;
			orientation.translation() += projectedGoalpostPosition;

			movedRootT.translation() -= projectedGoalpostPosition;
			movedRootT = Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY())*movedRootT;
			movedRootT.translation() += projectedGoalpostPosition;

			

			// std::cout<<"movedRootT.translation() : "<<movedRootT.translation().transpose()<<std::endl;
			// std::cout<<"orientation.translation() : "<<orientation.translation().transpose()<<std::endl;

			// orientation.linear() =

			// orientation.translation() -= displacementT.linear()* correctDisplacementT.translation();


			// orientation.translation() = getRootT(0).translation() + (getRootT(0)*correctDisplacementT).translation();
			// orientation.translation() = Eigen::Vector3d(3.0, 0.0, 0.0);

			// std::cout<<correctDisplacementT.translation().transpose()<<std::endl;
			// std::cout<<"aa.axis().transpose(): "<<aa.axis().transpose()<<std::endl;
			// std::cout<<"aa.angle(): "<<aa.angle()<<std::endl;

			Eigen::AngleAxisd temp(orientation.linear());

			dartPosition.segment(0,3) = temp.axis()*temp.angle();
			dartPosition.segment(3,3) = orientation.translation();

			tutorialTrajectory.push_back(dartPosition);
			tutorialControlVector.push_back(Utils::toEigenVec((xData[0][startFrame+j])));


			tutorialBallPosition.push_back(movedRootT*(ms->mPoses[startFrame+j]->ballPosition/100.0));
		}

		mTutorialTrajectories.push_back(tutorialTrajectory);
		mTutorialControlVectors.push_back(tutorialControlVector);
		mTutorialBallPositions.push_back(tutorialBallPosition);
	}

}




EnvironmentPackage::EnvironmentPackage(Environment* env, int numChars)
{
	this->numChars = numChars;
	this->bsm.resize(numChars);
	this->mCharacters.resize(numChars);
	for(int i=0;i<numChars;i++)
	{
		this->bsm[i] = new BStateMachine();
		this->mCharacters[i] = new Character3D("");
		SkeletonPtr bvhSkel = dart::utils::SkelParser::readSkeleton(env->mBvhParser->skelFilePath);
		env->addFingerSegmentToSkel(bvhSkel);
		this->mCharacters[i]->mSkeleton = bvhSkel;
	}
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
	this->mLFootDetached = env->mLFootDetached;
	this->mRFootDetached = env->mRFootDetached;
	this->mObstacles = env->mObstacles;

	this->mLFootContacting = env->mLFootContacting;
	this->mRFootContacting = env->mRFootContacting;
	this->mLLastFootPosition = env->mLLastFootPosition;
	this->mRLastFootPosition = env->mRLastFootPosition;

	this->mIsTerminalState = env->mIsTerminalState;
	this->mIsFoulState = env->mIsFoulState;
	this->mTimeElapsed = env->mTimeElapsed;
	this->mAccScore = env->mAccScore;
	this->mPrevActionTypes = env->mPrevActionTypes;
	this->prevFreeBallPositions = env->prevFreeBallPositions;
	this->gotReward = env->gotReward;


	this->ballSkelPosition = env->ballSkel->getPositions();

	for(int i=0;i<numChars;i++)
	{
		this->bsm[i]->copy(env->bsm[i]);
		this->mCharacters[i]->copy(env->mCharacters[i]);
	}
	env->mMgb->motionGenerators[0]->saveHiddenState(env->mBatchIndex);

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
	env->mLFootDetached = this->mLFootDetached;
	env->mRFootDetached = this->mRFootDetached;
	env->mObstacles = this->mObstacles;
	env->mCurHeadingAngle = this->mCurHeadingAngle;
	
	env->mLFootContacting = this->mLFootContacting;
	env->mRFootContacting = this->mRFootContacting;
	env->mLLastFootPosition = this->mLLastFootPosition;
	env->mRLastFootPosition = this->mRLastFootPosition;

	env->mIsTerminalState = this->mIsTerminalState;
	env->mIsFoulState = this->mIsFoulState;
	env->mTimeElapsed = this->mTimeElapsed;
	env->mAccScore = this->mAccScore;
	env->mPrevActionTypes = this->mPrevActionTypes;
	env->prevFreeBallPositions = this->prevFreeBallPositions;
	env->gotReward = this->gotReward;


	env->ballSkel->setPositions(this->ballSkelPosition);
	for(int i=0;i<numChars;i++)
	{
		env->bsm[i]->copy(this->bsm[i]);
		env->mCharacters[i]->copy(this->mCharacters[i]);

	}

	env->mMgb->motionGenerators[0]->restoreHiddenState(env->mBatchIndex);
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
	mPrevEnvSituation->restoreEnvironment(this);
}

void
Environment::
saveEnvironment()
{
	//Initial save
	if(curFrame == savedFrame)
	{
		// std::cout<<"BatchIndex : "<<mBatchIndex<<" Should not be saved"<<std::endl;
		return;
	}

	if(curFrame>1 && curFrame%typeFreq == 1)
	{
		if(mCurActionTypes[0] == 0)
		{
			mPrevEnvSituation->saveEnvironment(this);
			savedFrame = curFrame;
		}

	}
}

BStateMachine::BStateMachine()
{
	// curState = BasketballState::POSITIONING;
	// prevAction = 4;


	curState = BasketballState::DRIBBLING;
	prevAction = 0;
}

void
BStateMachine::
copy(BStateMachine *bsm)
{
	this->curState = bsm->curState;
	this->prevAction = bsm->prevAction;
}

int
BStateMachine::
transition(int action, bool transitState)
{
	if(curState == BasketballState::POSITIONING)
	{
		if(action == 4)
		{
			prevAction = action;
			return action;
		}
		else if(action == 2)
		{
			if(transitState)
				curState = BasketballState::DRIBBLING;
			prevAction = action;
			return 2;
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

// std::vector<int>
// BStateMachine::
// getAvailableActions()
// {
// 	if(curState == BasketballState::POSITIONING)
// 	{
// 		return std::vector<int>{2, 4, 5};
// 	}
// 	else if(curState == BasketballState::BALL_CATCH_1)
// 	{
// 		return std::vector<int>{0, 1, 3};//, 6, 7};
// 	}
// 	else if(curState == BasketballState::DRIBBLING)
// 	{
// 		return std::vector<int>{0, 1, 3};//, 6, 7};
// 	}
// 	else if(curState == BasketballState::BALL_CATCH_2)
// 	{
// 		return std::vector<int>{1, 3};//, 6, 7};

// 	}
// 	else
// 	{
// 		std::cout<<"Wrong state in basket ball state!"<<std::endl;
// 		exit(0);
// 	}
// }


std::vector<int>
BStateMachine::
getAvailableActions()
{
	if(curState == BasketballState::DRIBBLING)
	{
		return std::vector<int>{0, 1};//, 6, 7};
	}
	else if(curState == BasketballState::POSITIONING)
	{
		//Do not use now
		return std::vector<int>{2, 4};//, 6, 7};

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
	else if(curState == BasketballState::DRIBBLING)
	{
		return 1;
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
Environment::genObstacleNearGoalpost(double angle)
{
	double dRange = 1.0;

	double distance = 2.0;
	if(angle == -1)
		angle = (double) rand()/RAND_MAX * M_PI/1.0;// - M_PI/4.0;



	// std::cout<<angle<<std::endl;
	distance = (double) rand()/RAND_MAX * 4.0 + 2.0;

	Eigen::Vector3d obstaclePosition = mTargetBallPosition;
	obstaclePosition += distance * Eigen::Vector3d(cos(M_PI/2.0 + angle), 0.0, sin(M_PI/2.0 + angle));
	obstaclePosition[1] = 0.0;

	Eigen::Vector3d targetPlaneNormal = obstaclePosition - mCharacters[0]->getSkeleton()->getRootBodyNode()->getCOM();
	targetPlaneNormal[1] = 0.0;

	while(targetPlaneNormal.norm()>10.0)
	{
		if(angle == -1)
			angle = (double) rand()/RAND_MAX * M_PI/1.0;// - M_PI/4.0;



		// std::cout<<angle<<std::endl;
		distance = (double) rand()/RAND_MAX * 4.0 + 2.0;

		obstaclePosition = mTargetBallPosition;
		obstaclePosition += distance * Eigen::Vector3d(cos(M_PI/2.0 + angle), 0.0, sin(M_PI/2.0 + angle));
		obstaclePosition[1] = 0.0;

		targetPlaneNormal = obstaclePosition - mCharacters[0]->getSkeleton()->getRootBodyNode()->getCOM();
		targetPlaneNormal[1] = 0.0;
	}

	// std::cout<<"#######################################"<<std::endl;
	if(randomPointTrajectoryStart)
	{
		obstaclePosition[0] = 7.2;
		obstaclePosition[2] = 0.5;
	}


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