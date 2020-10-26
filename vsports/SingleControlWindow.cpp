#include "SingleControlWindow.h"
#include "../render/GLfunctionsDART.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include "../pyvs/EnvironmentPython.h"
#include <dart/dart.hpp>
#include <dart/utils/utils.hpp>
#include "../motion/BVHmanager.h"
#include <numeric>
// #include "./common/loadShader.h"
// #include <GL/glew.h>
#include <GL/glut.h>
// #include <GL/glew.h>
#include <GLFW/glfw3.h>
// Include GLM
#include <glm/glm.hpp>
#include "../extern/ICA/plugin/MotionGenerator.h"
#include "../extern/ICA/plugin/MotionGeneratorBatch.h"
#include "../extern/ICA/plugin/utils.h"
#include "../extern/ICA/Utils/PathManager.h"
#include "../extern/ICA/CharacterControl/MotionRepresentation.h"
#include "../utils/Utils.h"
using namespace glm;
#include <iostream>
#include <random>
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

// double floorDepth = -0.1;

void
SingleControlWindow::
initWindow(int _w, int _h, char* _name)
{
	mWindows.push_back(this);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE | GLUT_ACCUM);
	glutInitWindowPosition(100, 100);
	glutInitWindowSize(_w, _h);
	mWinIDs.push_back(glutCreateWindow(_name));
	// glutHideWindow();
	glutDisplayFunc(displayEvent);
	glutReshapeFunc(reshapeEvent);
	glutKeyboardFunc(keyboardEvent);
	glutKeyboardUpFunc(keyboardUpEvent);
	glutMouseFunc(mouseEvent);
	glutMotionFunc(motionEvent);
	glutTimerFunc(mDisplayTimeout, timerEvent, 0);
	mScreenshotTemp.resize(4*_w*_h);
	mScreenshotTemp2.resize(4*_w*_h);

}



SingleControlWindow::
SingleControlWindow()
:SimWindow(), mIsNNLoaded(false), mFrame(5273)
{

 	// srand (time(NULL));	
 	initCustomView();
	initGoalpost();

	// cout<<"1111"<<endl;
	mm = p::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = p::import("sys");
	// cout<<"222222"<<endl;

	boost::python::str module_dir = "../pyvs";
	sys_module.attr("path").attr("insert")(1, module_dir);
	p::exec("import os",mns);
	p::exec("import sys",mns);
	p::exec("import math",mns);
	p::exec("import sys",mns);
	// cout<<"3333333"<<endl;
	p::exec("import torch",mns);
	p::exec("import torch.nn as nn",mns);
	p::exec("import torch.optim as optim",mns);
	p::exec("import torch.nn.functional as F",mns);
	p::exec("import torchvision.transforms as T",mns);
	p::exec("import numpy as np",mns);
	p::exec("from Model import *",mns);
	p::exec("from VAE import VAEDecoder",mns);
	// p::exec("from Utils import RunningMeanStd",mns);
	// cout<<"444444"<<endl;
	controlOn = false;

	this->vsHardcoded = false;

	this->showCourtMesh = true;
// GLenum err = glewInit();
// if (err != GLEW_OK)
//   cout<<"Not ok with glew"<<endl; // or handle the error in a nicer way
// if (!GLEW_VERSION_2_1)  // check that the machine supports the 2.1 API.
//   cout<<"Not ok with glew version"<<endl; // or handle the error in a nicer way
//   cout<< glewGetString(err) <<endl; // or handle the error in a nicer way
	this->targetLocal.resize(10);
	this->targetLocal.setZero();
	this->goal.resize(2);
    this->goal.setZero();
    this->mTrackCharacter = true;

    for(int i=0;i<3;i++)
    {
        std::vector<Eigen::Isometry3d> prevHandTransform;
        prevHandTransform.push_back(Eigen::Isometry3d::Identity());
        prevHandTransform.push_back(Eigen::Isometry3d::Identity());
        this->prevHandTransforms.push_back(prevHandTransform);
    }

    fingerAngle = 0;
    fingerBallAngle =0;


}

SingleControlWindow::
SingleControlWindow(const char* nn_path,
					const char* control_nn_path)
:SingleControlWindow()
{
	int numActionTypes = 2;
	latentSize = 4;

	mEnv = new Environment(30, 180, 1, "../data/motions/basketData/motion/s_004_1_1.bvh", nn_path);
	reducedDim = false;
	mMotionGeneratorBatch = new ICA::dart::MotionGeneratorBatch(nn_path, mEnv->initDartNameIdMapping(), 1);

	mEnv->initialize(mMotionGeneratorBatch, 0);
	// mEnv->slaveReset();
	// mMotionGeneratorBatch->setCurrentDartPosition(mEnv->mCharacters[0]->getSkeleton()->getPositions(), 0);

	p::str str = ("num_state = "+std::to_string(mEnv->getNumState())).c_str();
	p::exec(str,mns);


	if(reducedDim)
		str = "num_action = 4";
	else
		str = ("num_action = "+std::to_string(mEnv->getNumAction()-1)).c_str();
	p::exec(str, mns);

	nn_module_0 = new boost::python::object[mEnv->mNumChars];
	nn_module_1 = new boost::python::object[mEnv->mNumChars];
	nn_module_2 = new boost::python::object[mEnv->mNumChars];
	p::object *load_0 = new p::object[mEnv->mNumChars];
	p::object *load_1 = new p::object[mEnv->mNumChars];
	p::object *load_2 = new p::object[mEnv->mNumChars];

	p::object *load_rms_0 = new p::object[mEnv->mNumChars];
	p::object *load_rms_1 = new p::object[mEnv->mNumChars];

	// reset_hidden = new boost::python::object[mEnv->mNumChars];

	for(int i=0;i<mEnv->mNumChars;i++)
	{
		nn_module_0[i] = p::eval(("ActorCriticNN(num_state, "+to_string(numActionTypes)+", 0.0, True).cuda()").data(), mns);
		load_0[i] = nn_module_0[i].attr("load");
		load_rms_0[i] = nn_module_0[i].attr("loadRMS");
	}
	for(int i=0;i<mEnv->mNumChars;i++)
	{
		nn_module_1[i] = p::eval(("ActorCriticNN(num_state+"+to_string(numActionTypes)+", "+to_string(latentSize)+").cuda()").data(), mns);
		load_1[i] = nn_module_1[i].attr("load");
		load_rms_1[i] = nn_module_1[i].attr("loadRMS");
	}
	// for(int i=0;i<mEnv->mNumChars;i++)
	// {
	// 	nn_module_2[i] = p::eval("ActorCriticNN(num_state+5+5, 6 ).cuda()", mns);
	// 	load_2[i] = nn_module_2[i].attr("load");
	// }



	// for(int i=0;i<mEnv->mNumChars;i++)
	// {
	// 	nn_module_0[i] = p::eval("ActorCriticNN(num_state, 4).cuda()", mns);
	// 	load_0[i] = nn_module_0[i].attr("load");
	// }
	// for(int i=0;i<mEnv->mNumChars;i++)
	// {
	// 	nn_module_1[i] = p::eval("ActorCriticNN(num_state+4, 2).cuda()", mns);
	// 	load_1[i] = nn_module_1[i].attr("load");
	// }


	load_0[0](string(control_nn_path) + "_0.pt");
	load_1[0](string(control_nn_path) + "_1.pt");

	std::string dir = control_nn_path;
	std::string subdir = "";
	while(dir.find("/")!=std::string::npos)
	{
		subdir += dir.substr(0,dir.find("/")+1);
		dir = dir.substr(dir.find("/")+1);
	}


	load_rms_0[0]((subdir + "rms.ms").data());
	load_rms_1[0]((subdir + "rms.ms").data());


	// load_2[0](string(control_nn_path) + "_2.pt");
	std::cout<<"Loaded control nn : "<<control_nn_path<<std::endl;


	nn_module_decoders = new boost::python::object[numActionTypes];
	p::object* load_decoders = new p::object[numActionTypes];

	for(int i=0;i<numActionTypes;i++)
	{
		nn_module_decoders[i] = p::eval("VAEDecoder().cuda()", mns);
		load_decoders[i] = nn_module_decoders[i].attr("load");
	}

	load_decoders[0]("../pyvs/vae_nn4/vae_action_decoder_"+to_string(0)+".pt");
	load_decoders[1]("../pyvs/vae_nn4/vae_action_decoder_"+to_string(3)+".pt");
	std::cout<<"Loaded VAE decoder"<<std::endl;



	mActions.resize(mEnv->mNumChars);
	for(int i=0;i<mActions.size();i++)
	{
		mActions[i].resize(mEnv->getNumAction());
		mActions[i].setZero();
	}
	mStates.resize(mEnv->mNumChars);

	targetActionType = 0;
	actionDelay = 0;
	// bvhParser = new BVHparser(bvh_path, BVHType::BASKET);
	// bvhParser->writeSkelFile();

	// cout<<bvhParser->skelFilePath<<endl;

	// SkeletonPtr bvhSkel = dart::utils::SkelParser::readSkeleton(bvhParser->skelFilePath);
	// charNames.push_back(getFileName_(bvh_path));
	// cout<<charNames[0]<<endl;	
	// BVHmanager::setPositionFromBVH(bvhSkel, bvhParser, 0);
	// mEnv->mWorld->addSkeleton(bvhSkel);


	// cout<<"Before MotionGenerator"<<endl;
	// exit(0);
	// cout<<"BVH skeleton dofs : "<<bvhSkel->getNumDofs()<<endl;
	// cout<<"BVH skeleton numBodies : "<<bvhSkel->getNumBodyNodes()<<endl;
	// initDartNameIdMapping();
	// mMotionGenerator = new ICA::dart::MotionGenerator(nn_path, this->dartNameIdMap);

	// cout<<bvhSkel->getPositions().transpose()<<endl;


	/// Read training X data
    std::string sub_dir = "data";
    std::string xDataPath=  PathManager::getFilePath_data(nn_path, sub_dir, "xData.dat", 0 );
    std::string xNormalPath=  PathManager::getFilePath_data(nn_path, sub_dir, "xNormal.dat");

    std::cout<<"xDataPath:"<<xDataPath<<std::endl;
    
    // if(! boost::filesystem::is_regular_file (xDataPath)) break;
    this->xData.push_back(MotionRepresentation::readXData(xNormalPath, xDataPath, sub_dir));

}



// void
// SingleControlWindow::
// initDartNameIdMapping()
// {    
// 	SkeletonPtr bvhSkel = mEnv->mWorld->getSkeleton(charNames[0]);
// 	int curIndex = 0;
// 	// cout<<bvhSkel->getNumBodyNodes()<<endl;
// 	for(int i=0;i<bvhSkel->getNumBodyNodes();i++)
// 	{
// 		this->dartNameIdMap[bvhSkel->getBodyNode(i)->getName()] = curIndex;
// 		curIndex += bvhSkel->getBodyNode(i)->getParentJoint()->getNumDofs(r);
// 	}

// 	// cout<<this->dartNameIdMap.size()<<endl;
// 	// for(auto& nameMap : this->dartNameIdMap)
// 	// {
// 	// 	cout<<nameMap.first<<" "<<nameMap.second<<endl;
// 	// }
// }

void
SingleControlWindow::
initCustomView()
{
	// mCamera->eye = Eigen::Vector3d(3.60468, -4.29576, 1.87037);
	// mCamera->lookAt = Eigen::Vector3d(-0.0936473, 0.158113, 0.293854);
	// mCamera->up = Eigen::Vector3d(-0.132372, 0.231252, 0.963847);
	mCamera->eye = Eigen::Vector3d(5.0, 8.0, 18.0);
	mCamera->lookAt = Eigen::Vector3d(5.0, 0.0, 0.0);
	mCamera->up = Eigen::Vector3d(0.0, 1.0, 0.0);

}

void
SingleControlWindow::
initGoalpost()
{
	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-8.0, 0.25 + floorDepth, 0.0), "red");
	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(8.0, 0.25 + floorDepth, 0.0), "blue");

	mWorld->addSkeleton(redGoalpostSkel);
	mWorld->addSkeleton(blueGoalpostSkel);
}



void
SingleControlWindow::
keyboard(unsigned char key, int x, int y)
{
	SkeletonPtr manualSkel = mEnv->getCharacter(0)->getSkeleton();

	switch(key)
	{
		case 'c':
			mTakeScreenShot = true;
			// cout<<mCamera->eye.transpose()<<endl;
			// cout<<mCamera->lookAt.transpose()<<endl;
			// cout<<mCamera->up.transpose()<<endl;
			break;
		case 'w':
            keyarr[int('w')] = PUSHED;
            break;
        case 's':
            keyarr[int('s')] = PUSHED;
            break;
        case 'a':
            keyarr[int('a')] = PUSHED;
            break;
        case 'd':
            keyarr[int('d')] = PUSHED;
            break;
        case 'j':
        	mEnv->genObstacleNearGoalpost();
        	break;
        case 'k':
        	mEnv->removeOldestObstacle();
        	break;
		case 't':
        {
            mTrackCharacter = !mTrackCharacter;
            break;
        }
		case 'r':
        {
            mEnv->slaveReset();
            break;
        }
		case 'h':
			mEnv->reset();
			// mEnv->getCharacter(1)->getSkeleton()->setPositions(Eigen::Vector2d(0.0, 0.0));
			// for(int i=0;i<4;i++){	
			// 	reset_hidden[i]();
			// }


			// reset_hidden[2]();
			// reset_hidden[3]();
			break;
		case 'g':
			mEnv->goBackEnvironment();
			// mEnv->getCharacter(1)->getSkeleton()->setPositions(Eigen::Vector2d(0.0, 0.0));
			// for(int i=0;i<4;i++){	
			// 	reset_hidden[i]();
			// }


			// reset_hidden[2]();
			// reset_hidden[3]();
			break;
		case 'l':
			controlOn = !controlOn;
			break;
		case 'i':
			showCourtMesh = !showCourtMesh;
			break;
		case 'p':
			// int numActions =6;
			int curAction;
		    for(int i=4;i<4+6;i++)
		    {
		        if(xData[0][mFrame+10][i] >= 0.5)
		            curAction = i-4;
		    }
		    while(curAction!=1 && curAction!=2 && curAction!=3)
		    {
		    	mFrame++;
		    	for(int i=4;i<4+6;i++)
			    {
			        if(xData[0][mFrame+10][i] >= 0.5)
			        {
			            curAction = i-4;
			        }
			    }
		    }
			break;
		case ']':
			mFrame += 100;
			step();
			break;
		case '[':
			mFrame -= 100;
			if(mFrame <0)
				mFrame = 0;
			break;

		default: SimWindow::keyboard(key, x, y);
	}
}
void
SingleControlWindow::
keyboardUp(unsigned char key, int x, int y)
{
	SkeletonPtr manualSkel = mEnv->getCharacter(0)->getSkeleton();

	switch(key)
	{
		case 'w':
			keyarr[int('w')] = NOTPUSHED;
			break;
		case 's':
			keyarr[int('s')] = NOTPUSHED;
			break;
		case 'a':
			keyarr[int('a')] = NOTPUSHED;
			break;
		case 'd':
			keyarr[int('d')] = NOTPUSHED;
			break;
		case 'g':
			keyarr[int('g')] = NOTPUSHED;
			break;
		case 'h':
			keyarr[int('h')] = NOTPUSHED;
			break;
		case 'q':
			keyarr[int('q')] = NOTPUSHED;
			break;
		case 'e':
			keyarr[int('e')] = NOTPUSHED;
			break;
	}
}
void
SingleControlWindow::
timer(int value)
{
	// time_check_start();
	if(mPlay)
		value = step();
	// display();
	// glutSwapBuffers();
	glutPostRedisplay();
	// time_check_end();
	SimWindow::timer(value);
}



void
SingleControlWindow::
applyKeyBoardEvent()
{
    double scale = 150.0;
    
    Eigen::Vector2d frontVec, rightVec;


    Eigen::Vector3d temp;
    temp = (mCamera->lookAt - mCamera->eye);
    frontVec[0] = temp[0];
    frontVec[1] = temp[2];

    frontVec.normalize();

    rightVec[0] = -frontVec[1];
    rightVec[1] = frontVec[0];

 
    if(keyarr[int('w')] == PUSHED)
        this->targetLocal.segment(0,2) += scale*frontVec;
    if(keyarr[int('s')] == PUSHED)
        this->targetLocal.segment(0,2) += -scale*frontVec;
    if(keyarr[int('a')] == PUSHED)
        this->targetLocal.segment(0,2) += -scale*rightVec;
    if(keyarr[int('d')] == PUSHED)
        this->targetLocal.segment(0,2) += scale*rightVec;
}
void
SingleControlWindow::
applyMouseEvent()
{
    Eigen::Vector3d facing = mCamera->lookAt - mCamera->eye;
    this->targetLocal[2] = facing[0];
    this->targetLocal[3] = facing[2];
    this->targetLocal.segment(2,2).normalize();
    // std::cout<<targetLocal.segment(2,2).transpose()<<std::endl;
}

int
SingleControlWindow::
step()
{
	std::chrono::time_point<std::chrono::system_clock> m_time_check_s = std::chrono::system_clock::now();
	if(mEnv->isTerminalState())// || mEnv->isFoulState())
	{
		sleep(1);
		mFrame = 5273;
		mEnv->slaveReset();
	}
	// std::cout<<"mFrame : "<<mFrame<<std::endl;
    // std::cout<<"RNN Time : "<<std::endl;
    // time_check_start();
	this->targetLocal.setZero();



	// std::cout<<mStates[0].transpose()<<std::endl;

	// mEnv->bsm[0]->curState = BasketballState::BALL_CATCH_1;
	// mActions[0] = Utils::toEigenVec(this->xData[0][mFrame]);

	
	// Eigen::VectorXd fullAction = Utils::toEigenVec(this->xData[0][mFrame]);



	// std::cout<<
	// mEnv->setActionType(0, getActionTypeFromVec(fullAction.segment(4,5))/3);

	// Eigen::VectorXd remainedAction(fullAction.rows()-5);
	// remainedAction.segment(0,4) = fullAction.segment(0,4);
	// remainedAction.segment(4,5) = fullAction.segment(4+5,5);

	// std::cout<<mEnv->mActions[0]<<std::endl;
	// std::cout<<Utils::toEigenVec(this->xData[0][mFrame]).transpose()<<std::endl;

	// time_check_start();


	getActionFromNN(0);


	// time_check_end();


	// time_check_start();

	// mEnv->setAction(0, remainedAction);
	mEnv->setAction(0, mActions[0]);
	// time_check_end();

	std::cout<<mEnv->mActions[0].segment(0,9).transpose()<<std::endl;
	std::cout<<mEnv->mActions[0].segment(9,5).transpose()<<std::endl;
	// std::cout<<mEnv->mActions[0].segment(16,4).transpose()<<std::endl;
	std::cout<<std::endl;


	
    // update prevHandTransform
    // updateHandTransform();

	// cout<<nextPosition.transpose()<<endl;
    // time_check_end();

    // std::cout<<"Simulator Time : "<<std::endl;
    // time_check_start();








	std::vector<std::vector<double>> concatControlVector;
	// std::cout<<"In SingleControlWindow :"<<std::endl;
	// for(int i=0;i<mEnv->slaveResetStateVector.size();i++)
	// {
	// 	std::cout<<mEnv->slaveResetStateVector[i]<<" ";
	// }
	// std::cout<<endl;
	// std::cout<<mEnv->slaveResetStateVector.transpose()<<std::endl;

	for(int id=0;id<1;++id)
	{
		if(mEnv->resetCount>30)
		{
			// std::cout<<"slave reset state vector"<<std::endl;
			mMotionGeneratorBatch->setBatchStateAndMotionGeneratorState(id, mEnv->slaveResetPositionVector);
		}
	}
	// std::cout<<"1111"<<std::endl;


	for(int id=0;id<1;++id)
	{
		if(mEnv->resetCount<0)
			concatControlVector.push_back(eigenToStdVec(mEnv->getMGAction(0)));
		else
			concatControlVector.push_back(eigenToStdVec(mEnv->slaveResetTargetVector));
	}
	// std::cout<<"2222"<<std::endl;

	// time_check_start();
	std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>>
	nextPoseAndContactsWithBatch = mMotionGeneratorBatch->generateNextPoseAndContactsWithBatch(concatControlVector);
	// time_check_end();

	// std::cout<<"3333"<<std::endl;


	// time_check_start();


	for(int id=0;id<1;++id)
	{
		mEnv->stepAtOnce(nextPoseAndContactsWithBatch[id]);
		// for(int j=0;j<num;j++)
		// 	this->step(id);
		// this->step
		// this->stepAtOnce(id);
	}






	// mEnv->stepAtOnce();
    // time_check_end();
    // std::cout<<std::endl;
	mEnv->getRewards();
	mFrame++;



	// glutPostRedisplay();
    std::chrono::duration<double> elapsed_seconds;
	elapsed_seconds = std::chrono::system_clock::now()-m_time_check_s;
    int calTime = std::min((int)(1000*elapsed_seconds.count()), 33);
	return calTime;
}

// void 
// SingleControlWindow::
// setBallPosition(bool leftContact)
// {
// 	SkeletonPtr bvhSkel = mEnv->mWorld->getSkeleton(charNames[0]);
//     Eigen::Isometry3d handTransform;
 
//     Eigen::VectorXd prevBallPosition = mEnv->ballSkel->getPositions();
 
//     if(leftContact)
//     {
//         handTransform = bvhSkel->getBodyNode("LeftHand")->getTransform();
//     }
//     else
//     {
//         handTransform = bvhSkel->getBodyNode("RightHand")->getTransform();
//     }



//     Eigen::VectorXd curBallPosition = mEnv->ballSkel->getPositions();
//     curBallPosition.segment(3,3) = handTransform * Eigen::Vector3d(0.10, 0.12, 0.0);
//     mEnv->ballSkel->setPositions(curBallPosition);
// }


// void 
// SingleControlWindow::
// setBallVelocity(bool leftContact)
// {
// 	SkeletonPtr bvhSkel = mEnv->mWorld->getSkeleton(charNames[0]);
//     Eigen::Isometry3d handTransform;
 
//     Eigen::VectorXd prevBallPosition = mEnv->ballSkel->getPositions();
 
//     if(leftContact)
//     {
//         handTransform = bvhSkel->getBodyNode("LeftHand")->getTransform();
//         prevBallPosition.segment(3,3) = prevHandTransforms[2][0] * Eigen::Vector3d(0.10, 0.12, 0.0);
//     }
//     else
//     {
//         handTransform = bvhSkel->getBodyNode("RightHand")->getTransform();
//         prevBallPosition.segment(3,3) = prevHandTransforms[2][1] * Eigen::Vector3d(0.10, 0.12, 0.0);
//         // handIndex = bvhSkel->getIndexOf(bvhSkel->getBodyNode("RightHand")->getParentJoint()->getDof(0));
//     }



//     Eigen::VectorXd curBallPosition = mEnv->ballSkel->getPositions();

//     if(mEnv->mTimeElapsed != 0)
//     {
//         mEnv->ballSkel->setVelocities((curBallPosition - prevBallPosition)*15.0);
//     }
// }

// void
// SingleControlWindow::
// updateHandTransform()
// {
// 	SkeletonPtr bvhSkel = mEnv->mWorld->getSkeleton(charNames[0]);
//     for(int i=2;i>0;i--)
//     {
//         this->prevHandTransforms[i] = this->prevHandTransforms[i-1];
//     }
//     std::vector<Eigen::Isometry3d> prevHandTransform;
//     prevHandTransform.push_back(bvhSkel->getBodyNode("LeftHand")->getTransform());
//     prevHandTransform.push_back(bvhSkel->getBodyNode("RightHand")->getTransform());

//     this->prevHandTransforms[0] = prevHandTransform;
// }

void
SingleControlWindow::
display()
{

    // time_check_end();
    // time_check_start();
	glClearColor(0.85, 0.85, 1.0, 1.0);
	// glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	initLights();
	
	std::vector<Character3D*> chars = mEnv->getCharacters();

	if(mTrackCharacter)
	{
		mCamera->lookAt = chars[0]->getSkeleton()->getCOM();
	}

	mCamera->apply();

	GUI::drawCoordinate(Eigen::Vector3d(0.0, 0.015, 0.0), 1.0);


	// for(int i=0;i<chars.size()-1;i++)
	// {
	// 	// if (i!=0)
	// 	// 	continue;
	// 	if(chars[i]->getTeamName() == "A")
	// 	{
	// 		// if(i==0)
	// 		// 	continue;
	// 		// cout<<mActions[i].transpose()<<endl;
	// 		if(mActions[i][2]>0)
	// 			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.8, 0.8));
	// 		else
	// 			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
	// 	}
	// 	else
	// 	{
	// 		if(mActions[i][2]>0)
	// 			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.8, 0.8, 1.0));
	// 		else
	// 			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));
	// 	}

	// }

	// cout<<mEnv->getLocalState(1)[_ID_SLOWED]<<endl;
	// cout<<mEnv->getLocalState(0).segment(_ID_GOALPOST_P+6, 2).transpose()<<endl;
	// cout<<endl;

	// mEnv->getCharacter(0)->getSkeleton()->getBodyNode("LeftFinger")->getParentJoint()->setPosition(0, fingerAngle);
	// mEnv->getCharacter(0)->getSkeleton()->getBodyNode("LeftFingerBall")->getParentJoint()->setPosition(0, fingerBallAngle);


	GUI::drawSkeleton(mEnv->floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5), showCourtMesh, false);

	GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.9, 0.6, 0.0));

	GUI::drawSkeleton(chars[0]->getSkeleton());

	// Eigen::Isometry3d rootIsometry = ICA::dart::getBaseToRootMatrix(mEnv->mMotionGenerator->motionGenerators[0]->mMotionSegment->getLastPose()->getRoot());
	// Eigen::Isometry3d rootIsometry = mEnv->mCharacters[0]->getSkeleton()->getRootBodyNode()->getWorldTransform();
	Eigen::Isometry3d rootIsometry = mEnv->getRootT(0);

	glPushMatrix();
	Eigen::Vector3d rootPosition = rootIsometry.translation();
	glTranslated(rootPosition[0], rootPosition[1], rootPosition[2]);
	Eigen::AngleAxisd rootAA(rootIsometry.linear());
	glRotated(180/M_PI*rootAA.angle(), rootAA.axis()[0], rootAA.axis()[1], rootAA.axis()[2]);
	// std::cout<<rootAA.angle()<<", "<<rootAA.axis().transpose()<<std::endl;
	GUI::drawCoordinate(Eigen::Vector3d::Zero(), 0.2);

	glPopMatrix();

	GUI::drawSphere(0.3, mEnv->mTargetBallPosition, Eigen::Vector3d(0.0, 0.0, 1.0));

	glPushMatrix();
	glTranslated(mEnv->mTargetBallPosition[0], 0, mEnv->mTargetBallPosition[2]);
	glBegin(GL_LINES);
    glColor3f(0.0, 0.0, 0.0);
    glVertex3f(0,0,0);
    glVertex3f(0,mEnv->mTargetBallPosition[1],0);
    glEnd();
    glPopMatrix();

    Eigen::Vector3d targetBall2DPosition = mEnv->mTargetBallPosition;
    targetBall2DPosition[1] = 0;
	GUI::drawSphere(0.05, targetBall2DPosition, Eigen::Vector3d(0.0, 0.0, 0.0));


	for(int i=0;i<mEnv->mObstacles.size();i++)
	{
		glPushMatrix();
		glTranslated(mEnv->mObstacles[i][0], 2.0, mEnv->mObstacles[i][2]);
		GUI::drawCylinder(0.5, 2.0, Eigen::Vector3d(0.3, 0.3, 0.3));
		glPopMatrix();
	}

	// cout<<"3333"<<endl;

	// std::string scoreString
	// = "Red : "+to_string((int)(mEnv->mAccScore[0] + mEnv->mAccScore[1]))+" |Blue : "+to_string((int)(mEnv->mAccScore[2]+mEnv->mAccScore[3]));

	// std::string scoreString
	// = "Red : "+to_string((mEnv->mAccScore[0]));

	//+" |Blue : "+to_string((int)(mEnv->mAccScore[1]));
	// = "Red : "+to_string((getRNDFeatureDiff(0)));//+" |Blue : "+to_string((int)(mEnv->mAccScore[1]));
	// cout<<"444444"<<endl;

	// cout<<mEnv->getCharacters()[0]->getSkeleton()->getVelocities().transpose()<<endl;
	// cout<<mActions[1][3]<<endl;

	GUI::drawStringOnScreen(0.2, 0.6, to_string((int)mEnv->mCharacters[0]->blocked), true, Eigen::Vector3d::Zero());

	GUI::drawStringOnScreen(0.8, 0.8, to_string(mEnv->getElapsedTime()), true, Eigen::Vector3d(0.5, 0.5, 0.5));

	if(mEnv->curContact[0]==0 || mEnv->curContact[0]==2)
	{
		Eigen::Isometry3d handIsometry = mEnv->mCharacters[0]->getSkeleton()->getBodyNode("LeftFinger")->getWorldTransform();

		glPushMatrix();
		Eigen::Vector3d handPosition = handIsometry.translation();
		glTranslated(handPosition[0], handPosition[1], handPosition[2]);
		Eigen::AngleAxisd handAA(handIsometry.linear());
		glRotated(180/M_PI*handAA.angle(), handAA.axis()[0], handAA.axis()[1], handAA.axis()[2]);
		// glTranslated(0.035, 0.0, 0.0);

		// GUI::drawSphere(0.1, mEnv->mCharacters[0]->getSkeleton()->getBodyNode("LeftHand")->getWorldTransform().translation(), Eigen::Vector3d::Zero());
		glColor3f(0.0, 0.0, 0.0);
		GUI::drawCube(Eigen::Vector3d(0.072, 0.035, 0.055));

		glPopMatrix();
	}
	if(mEnv->curContact[0]==1 || mEnv->curContact[0]==2)
	{
		Eigen::Isometry3d handIsometry = mEnv->mCharacters[0]->getSkeleton()->getBodyNode("RightFinger")->getWorldTransform();

		glPushMatrix();
		Eigen::Vector3d handPosition = handIsometry.translation();
		glTranslated(handPosition[0], handPosition[1], handPosition[2]);
		Eigen::AngleAxisd handAA(handIsometry.linear());
		glRotated(180/M_PI*handAA.angle(), handAA.axis()[0], handAA.axis()[1], handAA.axis()[2]);
		// glTranslated(0.035, 0.0, 0.0);

		// GUI::drawSphere(0.1, mEnv->mCharacters[0]->getSkeleton()->getBodyNode("LeftHand")->getWorldTransform().translation(), Eigen::Vector3d::Zero());
		glColor3f(0.0, 0.0, 0.0);
		GUI::drawCube(Eigen::Vector3d(0.072, 0.035, 0.055));

		glPopMatrix();	
	}


	if(mEnv->mLFootContacting[0])
	{
		Eigen::Isometry3d handIsometry = mEnv->mCharacters[0]->getSkeleton()->getBodyNode("LeftToe")->getWorldTransform();

		glPushMatrix();
		Eigen::Vector3d handPosition = handIsometry.translation();
		glTranslated(handPosition[0], handPosition[1], handPosition[2]);
		Eigen::AngleAxisd handAA(handIsometry.linear());
		glRotated(180/M_PI*handAA.angle(), handAA.axis()[0], handAA.axis()[1], handAA.axis()[2]);
		glTranslated(0.05, 0.0, 0.0);

		// GUI::drawSphere(0.1, mEnv->mCharacters[0]->getSkeleton()->getBodyNode("LeftHand")->getWorldTransform().translation(), Eigen::Vector3d::Zero());
		glColor3f(0.3, 0.3, 0.3);
		GUI::drawCube(Eigen::Vector3d(0.10, 0.06, 0.06));

		glPopMatrix();	
	}

	if(mEnv->mRFootContacting[0])
	{
		Eigen::Isometry3d handIsometry = mEnv->mCharacters[0]->getSkeleton()->getBodyNode("RightToe")->getWorldTransform();

		glPushMatrix();
		Eigen::Vector3d handPosition = handIsometry.translation();
		glTranslated(handPosition[0], handPosition[1], handPosition[2]);
		Eigen::AngleAxisd handAA(handIsometry.linear());
		glRotated(180/M_PI*handAA.angle(), handAA.axis()[0], handAA.axis()[1], handAA.axis()[2]);
		glTranslated(0.05, 0.0, 0.0);

		// GUI::drawSphere(0.1, mEnv->mCharacters[0]->getSkeleton()->getBodyNode("LeftHand")->getWorldTransform().translation(), Eigen::Vector3d::Zero());
		glColor3f(0.3, 0.3, 0.3);
		GUI::drawCube(Eigen::Vector3d(0.10, 0.06, 0.06));

		glPopMatrix();	
	}


	// GUI::drawVerticalLine(this->goal.segment(0,2), Eigen::Vector3d(1.0, 1.0, 1.0));

	int numActions = 5;

    std::string curAction;

    bool useXData = false;
    if(useXData)
    {
	    for(int i=4;i<4+numActions;i++)
	    {
	        if(xData[0][mFrame][i] >= 0.5)
	            curAction = std::to_string(i-4);
	    }
	    curAction = curAction+"     "+std::to_string(xData[0][mFrame][4+numActions+4]/30.0);
    }
    else
    {

	    int maxIndex = 0;
	    double maxValue = -100;
	    for(int i=4;i<4+numActions;i++)
	    {
	        if(mEnv->mActions[0][i]> maxValue)
	        {
	        	maxValue= mEnv->mActions[0][i];
	        	maxIndex = i;
	        }
	    }
	    curAction = std::to_string(maxIndex-4);
	    curAction = curAction+"     "+std::to_string(mEnv->mActions[0][4+numActions+4]/30.0);
    }
	if(mEnv->mCurActionTypes[0] == 1 || mEnv->mCurActionTypes[0] == 3 )
	{
		Eigen::Vector3d ballTargetPosition = Eigen::Vector3d(0.0, mEnv->mActions[0][4+numActions+3]/100.0, 0.0);
		Eigen::Vector3d ballTargetVelocity = mEnv->mActions[0].segment(4+numActions,3)/100.0;
		ballTargetPosition = rootIsometry * ballTargetPosition;
		ballTargetVelocity = rootIsometry.linear() * ballTargetVelocity;

		// std::cout<<ballTargetPosition.transpose()<<std::endl;
		// std::cout<<ballTargetVelocity.transpose()<<std::endl;
		// std::cout<<std::endl;
		GUI::drawArrow3D(ballTargetPosition, ballTargetVelocity, ballTargetVelocity.norm()/8.0, 0.05, Eigen::Vector3d(1.0, 0.0, 0.0), 0.08);
	}




    GUI::drawStringOnScreen(0.2, 0.25, std::to_string(mEnv->mActions[0][4+numActions+4]/30.0), true, Eigen::Vector3d(1,1,1));

    std::string score = "Score : "+to_string(mEnv->mAccScore[0]);
   
	GUI::drawBoxOnScreen(0.25, 0.75, Eigen::Vector2d(15.0, 4.0),Eigen::Vector3d(1.0, 1.0, 1.0), true);
    GUI::drawStringOnScreen(0.2, 0.75, score, true, Eigen::Vector3d(0.0,0.0,0.0));


    GUI::drawStringOnScreen(0.2, 0.55, std::to_string(mEnv->mCurBallPossessions[0]), true, Eigen::Vector3d(1,1,1));

	showAvailableActions();

	// std::cout<<mEnv->mCurActionTypes[0]<<std::endl;
	GUI::drawBoxOnScreen(0.2+0.6*((double)mEnv->mCurActionTypes[0] / (numActions)), 0.2, Eigen::Vector2d(6.0, 4.0),Eigen::Vector3d(1.0, 1.0, 1.0));


	glutSwapBuffers();
	if(mTakeScreenShot)
	{
		screenshot();
	}
	// glutPostRedisplay();
}

std::string
SingleControlWindow::
indexToStateString(int index)
{
	switch(index)
	{
		case 0:
			return "P_x";
		case 1:
			return "P_y";
		case 2:
			return "V_x";
		case 3:
			return "V_y";
		case 4:
			return "BP_x";
		case 5:
			return "BP_y";
		case 6:
			return "BV_x";
		case 7:
			return "BV_y";
		case 8:
			return "Kick";
		case 9:
			return "B_GP";
		case 10:
			return " ";
		case 11:
			return "B_GP";
		case 12:
			return " ";
		case 13:
			return "R_GP";
		case 14:
			return " ";
		case 15:
			return "R_GP";
		case 16:
			return " ";
		default:
			return "N";
	}
	return "N";
}

void
SingleControlWindow::
mouse(int button, int state, int x, int y) 
{
    mPrevX = x;
    mPrevY = y;
    if(button == GLUT_LEFT_BUTTON && glutGetModifiers()==GLUT_ACTIVE_SHIFT){
        GLdouble modelview[16], projection[16];
        GLint viewport[4];

        double height = glutGet(GLUT_WINDOW_HEIGHT);

        glGetDoublev(GL_MODELVIEW_MATRIX, modelview);
        glGetDoublev(GL_PROJECTION_MATRIX, projection);
        glGetIntegerv(GL_VIEWPORT, viewport);

        double objx1, objy1, objz1;
        double objx2, objy2, objz2;

        int res1 = gluUnProject(x, height - y, 0, modelview, projection, viewport, &objx1, &objy1, &objz1);
        int res2 = gluUnProject(x, height - y, 10, modelview, projection, viewport, &objx2, &objy2, &objz2);

        this->goal[0] = objx1 + (objx2 - objx1)*(objy1)/(objy1-objy2);
        this->goal[1] = objz1 + (objz2 - objz1)*(objy1)/(objy1-objy2);

        // this->goal.segment(2,24).setZero();
        // // this->goal[25] = -1;
        // // this->goal[3] = 1;

        // mMotionGenerator->mGoal = this->goal;
        // mMotionGenerator->mGoal.segment(0,2) *= 100.0;

       	// std::cout<<"new goal: "<<this->targetLocal..transpose()<<std::endl;
	}
	else
	{
		SimWindow::mouse(button, state, x, y);
	}
}


void
SingleControlWindow::
motion(int x, int y)
{
	SimWindow::motion(x, y);
}

Eigen::VectorXd
SingleControlWindow::
toOneHotVector(Eigen::VectorXd action)
{
    int maxIndex = 0;
    double maxValue = -100;
    for(int i=0;i<action.size();i++)
    {
        if(action[i]> maxValue)
        {
        	maxValue= action[i];
        	maxIndex = i;
        }
    }
    Eigen::VectorXd result(action.size());
    result.setZero();
    result[maxIndex] = 1.0;
    return result;
}


Eigen::VectorXd
SingleControlWindow::
toOneHotVectorWithConstraint(int index, Eigen::VectorXd action)
{
	// std::cout<<"acition type "<<action.transpose()<<std::endl;
    int maxIndex = 0;
    double maxValue = -100;
    for(int i=0;i<action.size();i++)
    {
        if(action[i]> maxValue)
        {
        	maxValue= action[i];
        	maxIndex = i;
        }
    }
    maxIndex = mEnv->setActionType(index, maxIndex);
    Eigen::VectorXd result(action.size());
    result.setZero();
    result[maxIndex] = 1.0;
    return result;
}


int
SingleControlWindow::
getActionTypeFromVec(Eigen::VectorXd action)
{
    int maxIndex = 0;
    double maxValue = -100;
    for(int i=0;i<action.size();i++)
    {
        if(action[i]> maxValue)
        {
        	maxValue= action[i];
        	maxIndex = i;
        }
    }

    return maxIndex;
}



void
SingleControlWindow::
getActionFromNN(int index)
{
	p::object get_action_0;

	Eigen::VectorXd state = mEnv->getState(index);
	std::cout<<"action mask : "<<state.segment(state.size()-2, 2).transpose()<<std::endl;

	int numActions = 2;
	// int latentSize = 4;
	// std::cout<<state.segment(155,6).transpose()<<std::endl;
	// std::cout<<state.segment(mEnv->mCharacters[0]->getSkeleton()->getNumDofs(),12).transpose()<<std::endl;

	Eigen::VectorXd mActionType(numActions);
	mActionType.setZero();

	get_action_0 = nn_module_0[index].attr("get_action");

	p::tuple shape = p::make_tuple(state.size());
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape, dtype);

	float* dest = reinterpret_cast<float*>(state_np.get_data());
	for(int j=0;j<state.size();j++)
	{
		dest[j] = state[j];
	}

	p::object temp = get_action_0(state_np);
	np::ndarray action_np = np::from_object(temp);
	float* srcs = reinterpret_cast<float*>(action_np.get_data());

	for(int j=0;j<numActions;j++)
	{
		mActionType[j] = srcs[j];
	}

	// std::cout<<"mActionType : "<<mActionType.transpose()<<std::endl;
	std::cout<<"mEnv->curFrame : "<<mEnv->curFrame<<std::endl;
	if(mEnv->curFrame%10 == 1)
		mActionType = toOneHotVectorWithConstraint(index, mActionType);
	else
	{
		mActionType.setZero();
		int prevActionType = mEnv->mCurActionTypes[index]/3;
		mActionType[prevActionType] = 1.0;

	}

	int actionType = getActionTypeFromVec(mActionType);

	// mEnv->setActionType(index, actionType);


	///////////////
	Eigen::VectorXd mAction(mEnv->getNumAction() - numActions);
	Eigen::VectorXd state_1(state.size()+numActions);
	state_1.segment(0,state.size()) = state;
	state_1.segment(state.size(),numActions) = mActionType;

	p::object get_action_1;

	get_action_1 = nn_module_1[index].attr("get_action");

	p::tuple shape_1 = p::make_tuple(state_1.size());
	np::ndarray state_np_1 = np::empty(shape_1, dtype);

	float* dest_1 = reinterpret_cast<float*>(state_np_1.get_data());
	for(int j=0;j<state_1.size();j++)
	{
		dest_1[j] = state_1[j];
	}

	temp = get_action_1(state_np_1);
	np::ndarray action_np_1 = np::from_object(temp);
	float* srcs_1 = reinterpret_cast<float*>(action_np_1.get_data());

	for(int j=0;j<latentSize;j++)
	{
		mAction[j] = srcs_1[j];
	}

	///////////////
	// exit(0);
	// mAction.segment(9,2) = mAction.segment(4,2);


	Eigen::VectorXd encodedAction(latentSize);
	encodedAction = mAction.segment(0,encodedAction.size());
	Eigen::VectorXd decodedAction(9);

	// encodedAction.setOnes();
	// encodedAction *= -5.0;

	// encodedAction << 0.3710, -0.2950,  0.3723, -0.0413, -1.4446, -0.7784;

	p::object decode;

	decode = nn_module_decoders[actionType].attr("decodeAction");

	p::tuple shape_d = p::make_tuple(encodedAction.size());
	np::ndarray state_np_d = np::empty(shape_d, dtype);

	float* dest_d = reinterpret_cast<float*>(state_np_d.get_data());
	for(int j=0;j<encodedAction.size();j++)
	{
		dest_d[j] = encodedAction[j];
	}

	temp = decode(state_np_d);
	np::ndarray action_np_d = np::from_object(temp);
	float* src_d = reinterpret_cast<float*>(action_np_d.get_data());

	for(int j=0;j<9;j++)
	{
		decodedAction[j] = src_d[j];
	}

	std::cout<<"Cur Action Type : "<<actionType<<std::endl;
	std::cout<<"Encoded Action : "<<encodedAction.transpose()<<std::endl;
	std::cout<<"Decoded Action : "<<decodedAction.transpose()<<std::endl;



/*	Eigen::VectorXd mActionHandContact(6);
	Eigen::VectorXd state_2(state_1.size()+encodedAction.rows());
	state_2.segment(0,state_1.size()) = state_1;
	state_2.segment(state_1.size(),encodedAction.rows()) = encodedAction;

	p::object get_action_2;

	get_action_2 = nn_module_2[index].attr("get_action");

	p::tuple shape_2 = p::make_tuple(state_2.size());
	np::ndarray state_np_2 = np::empty(shape_2, dtype);

	float* dest_2 = reinterpret_cast<float*>(state_np_2.get_data());
	for(int j=0;j<state_2.size();j++)
	{
		dest_2[j] = state_2[j];
	}

	temp = get_action_2(state_np_2);
	np::ndarray action_np_2 = np::from_object(temp);
	float* srcs_2 = reinterpret_cast<float*>(action_np_2.get_data());

	for(int j=0;j<mActionHandContact.size();j++)
	{
		mActionHandContact[j] = srcs_2[j];
	}
*/
	// mAction.segment(0, decodedAction.size()) = decodedAction;
	// mAction.segment(decodedAction.size(),mActionHandContact.size()) = mActionHandContact;







	// std::cout<<"Decoded Action :"<<std::endl;
	// std::cout<<mAction.transpose()<<std::endl;


	// mAction = mEnv->mNormalizer->denormalizeAction(mAction);
	// std::cout<<mAction.segment(0,4).transpose()<<std::endl;
	// std::cout<<mAction.segment(4,8).transpose()<<std::endl;
	// std::cout<<mAction.segment(12,7).transpose()<<std::endl;
	// std::cout<<std::endl;
	// std::cout<<"-------------"<<std::endl;	
	mActions[index] = mEnv->mNormalizer->denormalizeAction(decodedAction);
	// mActions[index] = mAction;

	// return mActions[index];
}



/*
void
SingleControlWindow::
getActionFromNN(int index)
{
	p::object get_action_0;

	Eigen::VectorXd state = mEnv->getState(index);

	int numActions = 5;
	// std::cout<<state.segment(155,6).transpose()<<std::endl;
	// std::cout<<state.segment(mEnv->mCharacters[0]->getSkeleton()->getNumDofs(),12).transpose()<<std::endl;

	Eigen::VectorXd mAction(mEnv->getNumAction()-1);
	mAction.setZero();

	get_action_0 = nn_module_0[index].attr("get_action");

	p::tuple shape = p::make_tuple(state.size());
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape, dtype);

	float* dest = reinterpret_cast<float*>(state_np.get_data());
	for(int j=0;j<state.size();j++)
	{
		dest[j] = state[j];
	}

	p::object temp = get_action_0(state_np);
	np::ndarray action_np = np::from_object(temp);
	float* srcs = reinterpret_cast<float*>(action_np.get_data());

	for(int j=0;j<numActions;j++)
	{
		mAction[j] = srcs[j];
	}

	mAction.segment(0,numActions) = toOneHotVector(mAction.segment(0,numActions));

	///////////////

	Eigen::VectorXd state_1(state.size()+numActions);
	state_1.segment(0,state.size()) = state;
	state_1.segment(state.size(),numActions) = mAction.segment(0,numActions);

	p::object get_action_1;

	get_action_1 = nn_module_1[index].attr("get_action");

	p::tuple shape_1 = p::make_tuple(state_1.size());
	np::ndarray state_np_1 = np::empty(shape_1, dtype);

	float* dest_1 = reinterpret_cast<float*>(state_np_1.get_data());
	for(int j=0;j<state_1.size();j++)
	{
		dest_1[j] = state_1[j];
	}

	temp = get_action_1(state_np_1);
	np::ndarray action_np_1 = np::from_object(temp);
	float* srcs_1 = reinterpret_cast<float*>(action_np_1.get_data());

	for(int j=numActions;j<numActions+4;j++)
	{
		mAction[j] = srcs_1[j-numActions];
	}

	///////////////

	Eigen::VectorXd state_2(state.size()+numActions+4);
	state_2.segment(0,state_1.size()) = state_1;
	state_2.segment(state_1.size(),4) = mAction.segment(numActions,4);

	p::object get_action_2;

	get_action_2 = nn_module_2[index].attr("get_action");

	p::tuple shape_2 = p::make_tuple(state_2.size());
	np::ndarray state_np_2 = np::empty(shape_2, dtype);

	float* dest_2 = reinterpret_cast<float*>(state_np_2.get_data());
	for(int j=0;j<state_2.size();j++)
	{
		dest_2[j] = state_2[j];
	}

	temp = get_action_2(state_np_2);
	np::ndarray action_np_2 = np::from_object(temp);
	float* srcs_2 = reinterpret_cast<float*>(action_np_2.get_data());

	for(int j=numActions+4;j<mAction.size();j++)
	{
		mAction[j] = srcs_2[j-4-numActions];
	}



	// mAction = mEnv->mNormalizer->denormalizeAction(mAction);
	// std::cout<<mAction.segment(0,4).transpose()<<std::endl;
	// std::cout<<mAction.segment(4,8).transpose()<<std::endl;
	// std::cout<<mAction.segment(12,7).transpose()<<std::endl;
	// std::cout<<std::endl;
	// std::cout<<"-------------"<<std::endl;	
	mActions[index] = mEnv->mNormalizer->denormalizeAction(mAction);
}

*/
/*void
SingleControlWindow::
getActionFromNN(int index)
{
	p::object get_action_0;

	Eigen::VectorXd state = mEnv->getState(index);

	int numActions = 4;
	// std::cout<<state.segment(155,6).transpose()<<std::endl;
	// std::cout<<state.segment(mEnv->mCharacters[0]->getSkeleton()->getNumDofs(),12).transpose()<<std::endl;

	Eigen::VectorXd mAction(mEnv->getNumAction());
	mAction.setZero();

	get_action_0 = nn_module_0[index].attr("get_action");

	p::tuple shape = p::make_tuple(state.size());
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape, dtype);

	float* dest = reinterpret_cast<float*>(state_np.get_data());
	for(int j=0;j<state.size();j++)
	{
		dest[j] = state[j];
	}

	p::object temp = get_action_0(state_np);
	np::ndarray action_np = np::from_object(temp);
	float* srcs = reinterpret_cast<float*>(action_np.get_data());

	for(int j=0;j<numActions;j++)
	{
		mAction[j] = srcs[j];
	}



	// mAction.segment(0,numActions) = toOneHotVector(mAction.segment(0,numActions));

	///////////////

	Eigen::VectorXd state_1(state.size()+numActions);
	state_1.segment(0,state.size()) = state;
	state_1.segment(state.size(),numActions) = mAction.segment(0,numActions);

	p::object get_action_1;

	get_action_1 = nn_module_1[index].attr("get_action");

	p::tuple shape_1 = p::make_tuple(state_1.size());
	np::ndarray state_np_1 = np::empty(shape_1, dtype);

	float* dest_1 = reinterpret_cast<float*>(state_np_1.get_data());
	for(int j=0;j<state_1.size();j++)
	{
		dest_1[j] = state_1[j];
	}

	temp = get_action_1(state_np_1);
	np::ndarray action_np_1 = np::from_object(temp);
	float* srcs_1 = reinterpret_cast<float*>(action_np_1.get_data());

	for(int j=numActions;j<numActions+4;j++)
	{
		mAction[j] = srcs_1[j-numActions];
	}

	///////////////
	// exit(0);
	mAction.segment(14,2) = mAction.segment(4,2);


	Eigen::VectorXd encodedAction(4);
	encodedAction = mAction.segment(0,encodedAction.size());

	p::object decode;

	decode = nn_module_decoder.attr("decodeAction");

	p::tuple shape_d = p::make_tuple(encodedAction.size());
	np::ndarray state_np_d = np::empty(shape_d, dtype);

	float* dest_d = reinterpret_cast<float*>(state_np_d.get_data());
	for(int j=0;j<encodedAction.size();j++)
	{
		dest_d[j] = encodedAction[j];
	}

	temp = decode(state_np_d);
	np::ndarray action_np_d = np::from_object(temp);
	float* src_d = reinterpret_cast<float*>(action_np_d.get_data());

	for(int j=0;j<14;j++)
	{
		mAction[j] = src_d[j];
	}
	// std::cout<<"Decoded Action :"<<std::endl;
	// std::cout<<mAction.transpose()<<std::endl;


	// mAction = mEnv->mNormalizer->denormalizeAction(mAction);
	// std::cout<<mAction.segment(0,4).transpose()<<std::endl;
	// std::cout<<mAction.segment(4,8).transpose()<<std::endl;
	// std::cout<<mAction.segment(12,7).transpose()<<std::endl;
	// std::cout<<std::endl;
	// std::cout<<"-------------"<<std::endl;	
	mActions[index] = mEnv->mNormalizer->denormalizeAction(mAction);
}

*/

void
SingleControlWindow::
showAvailableActions()
{
	std::vector<int> aa = mEnv->bsm[0]->getAvailableActions();
	int numActionTypes = 5;
	for(int i=0;i<numActionTypes;i++)
	{
		// if( i == mEnv->mCurActionTypes[0])
		// 	GUI::drawStringOnScreen_Big(-0.01 + 0.2+0.6*(((double) i)/numActionTypes), 0.2, mEnv->actionNameMap[i], Eigen::Vector3d::UnitZ());
		// else 
		if(std::find(aa.begin(), aa.end(), i) != aa.end())
			GUI::drawStringOnScreen_Big(-0.01 + 0.2+0.6*(((double) i)/numActionTypes), 0.2, mEnv->actionNameMap[i], Eigen::Vector3d::Ones());
		else
			GUI::drawStringOnScreen_Big(-0.01 + 0.2+0.6*(((double) i)/numActionTypes), 0.2, mEnv->actionNameMap[i], Eigen::Vector3d(0.5, 0.5, 0.5));

	}
}


// void
// SingleControlWindow::
// getActionFromNN(int index)
// {
// 	p::object get_action;

// 	Eigen::VectorXd state = mEnv->getNormalizedState(index);
// 	// Eigen::VectorXd normalizedState = mNormalizer->normalizeState(state);
// 	// std::cout<<normalizedState.transpose()<<std::endl;
// 	// std::cout<<mEnv->mStates[0].segment(8,3).transpose()<<" // "<<mEnv->mStates[0].segment(146,3).transpose()<<std::endl;

// 	Eigen::VectorXd mAction(mEnv->getNumAction());
// 	mAction.setZero();

// 	get_action = nn_module[index].attr("get_action");

// 	p::tuple shape = p::make_tuple(state.size());
// 	np::dtype dtype = np::dtype::get_builtin<float>();
// 	np::ndarray state_np = np::empty(shape, dtype);

// 	float* dest = reinterpret_cast<float*>(state_np.get_data());
// 	for(int j=0;j<state.size();j++)
// 	{
// 		dest[j] = state[j];
// 	}

// 	p::object temp = get_action(state_np);
// 	np::ndarray action_np = np::from_object(temp);
// 	float* srcs = reinterpret_cast<float*>(action_np.get_data());



// 	if(reducedDim)
// 	{
// 		for(int j=0;j<4;j++)
// 		{
// 			mAction[j] = srcs[j];
// 		}
// 	}
// 	else
// 	{
// 		for(int j=0;j<mAction.rows();j++)
// 		{
// 			mAction[j] = srcs[j];
// 		}
// 	}

// 	mAction = mNormalizer->denormalizeAction(mAction);
// 	// std::cout<<mAction.transpose()<<std::endl;


// 	// mAction = mNormalizer->denormalizeAction(mAction);
// 	// std::


// 	// cout<<"Here?"<<endl;
// 	mActions[index] = mAction;
// 	// cout<<"NO"
// }

int getActionTypeFromVec(std::vector<double> vec)
{
	int maxValue = -100;
	int maxIndex = 0;
	for(int i=4;i<4+6;i++)
	{
		if(vec[i]>maxValue)
		{
			maxIndex = i;
			maxValue = vec[i];
		}
	}
	return maxIndex-4;
}

// void
// SingleControlWindow::
// getControlMeanStdByActionType(int actionType)
// {
// 	// std::cout<<this->xData[0].size()<<std::endl;
// 	std::vector<std::vector<double>> total;
// 	for(int i=0;i<this->xData[0].size();i++)
// 	{
// 		int curActionType = getActionTypeFromVec(this->xData[0][i]);

// 		if(curActionType == actionType)
// 		{
// 			total.push_back(this->xData[0][i]);
// 		}
// 	}

// 	std::vector<std::vector<double>> totalTranspose;

// 	// std::cout<<total[0].size()<<std::endl;
// 	for(int j=0;j<total[0].size();j++)
// 	{
// 		std::vector<double> row;
// 		for(int i=0;i<total.size();i++)
// 		{
// 			// std::cout<<i<<" "<<j<<std::endl;
// 			row.push_back(total[i][j]);
// 		}
// 		totalTranspose.push_back(row);
// 	}

// 	for(int i=0;i<totalTranspose.size();i++)
// 	{
// 		std::cout<<i<<" th :";
// 		double sum = std::accumulate(totalTranspose[i].begin(), totalTranspose[i].end(), 0.0);
// 		double mean = sum/totalTranspose[i].size();

// 		std::vector<double> diff(totalTranspose[i].size());
// 		std::transform(totalTranspose[i].begin(), totalTranspose[i].end(), diff.begin(),
// 		               std::bind2nd(std::minus<double>(), mean));

// 		double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
// 		double stdev = std::sqrt(sq_sum/(diff.size()-1));
// 		std::cout<<mean<<", "<<stdev<<" / "<<mean+2*stdev<<", "<<mean-2*stdev<<std::endl;
// 	}
// 	// exit(0);


// 	// double sum = std::accumulate(total.begin(), _InputIterator __last, _Tp __init)

// }