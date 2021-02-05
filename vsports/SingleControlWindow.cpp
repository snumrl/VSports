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
:SimWindow(), mIsNNLoaded(false), mFrame(0), windowFrame(0)
{
 	initCustomView();
	initGoalpost();

	mm = p::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = p::import("sys");

	boost::python::str module_dir = "../pyvs";
	sys_module.attr("path").attr("insert")(1, module_dir);
	p::exec("import os",mns);
	p::exec("import sys",mns);
	p::exec("import math",mns);
	p::exec("import sys",mns);
	p::exec("import torch",mns);
	p::exec("import torch.nn as nn",mns);
	p::exec("import torch.optim as optim",mns);
	p::exec("import torch.nn.functional as F",mns);
	p::exec("import torchvision.transforms as T",mns);
	p::exec("import numpy as np",mns);
	p::exec("from Model import *",mns);
	p::exec("from VAE import VAEDecoder",mns);

	controlOn = false;

	this->vsHardcoded = false;

	this->showCourtMesh = true;

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

    for(int i=0;i<2;i++)
    {
    	prevValues.push_back(0);
    	prevRewards.push_back(0);
    	curValues.push_back(0);
    }

}

SingleControlWindow::
SingleControlWindow(const char* nn_path,
					const char* control_nn_path)
:SingleControlWindow()
{
	int numActionTypes = 5;
	latentSize = 6;

	mEnv = new Environment(30, 180, 1, "../data/motions/basketData/motion/s_004_1_1.bvh", nn_path);
	reducedDim = false;
	mMotionGeneratorBatch = new ICA::dart::MotionGeneratorBatch(nn_path, mEnv->initDartNameIdMapping(), 1);

	mEnv->initialize(mMotionGeneratorBatch, 0);

	recorder = new Recorder();

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

	for(int i=0;i<mEnv->mNumChars;i++)
	{
		nn_module_0[i] = p::eval(("ActorCriticNN(num_state, "+to_string(numActionTypes)+", 0.0, True, True).cuda()").data(), mns);
		load_0[i] = nn_module_0[i].attr("load");
		load_rms_0[i] = nn_module_0[i].attr("loadRMS");
	}
	for(int i=0;i<mEnv->mNumChars;i++)
	{
		nn_module_1[i] = p::eval(("ActorCriticNN(num_state + "+to_string(numActionTypes)+", "+to_string(latentSize)+").cuda()").data(), mns);
		load_1[i] = nn_module_1[i].attr("load");
		load_rms_1[i] = nn_module_1[i].attr("loadRMS");
	}

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


	std::cout<<"Loaded control nn : "<<control_nn_path<<std::endl;


	nn_module_decoders = new boost::python::object[numActionTypes];
	p::object* load_decoders = new p::object[numActionTypes];

	for(int i=0;i<numActionTypes;i++)
	{
		nn_module_decoders[i] = p::eval("VAEDecoder().cuda()", mns);
		load_decoders[i] = nn_module_decoders[i].attr("load");
	}

	for(int i=0;i<numActionTypes;i++)
	{
		load_decoders[i]("../pyvs/vae_nn_sep_4/vae_action_decoder_"+to_string(i)+".pt");
	}
	// load_decoders[0]("../pyvs/vae_nn_sep/vae_action_decoder_"+to_string(0)+".pt");
	// load_decoders[1]("../pyvs/vae_nn_sep/vae_action_decoder_"+to_string(3)+".pt");
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


	/// Read training X data
    std::string sub_dir = "data";
    std::string xDataPath=  PathManager::getFilePath_data(nn_path, sub_dir, "xData.dat", 0 );
    std::string xNormalPath=  PathManager::getFilePath_data(nn_path, sub_dir, "xNormal.dat");

    std::cout<<"xDataPath:"<<xDataPath<<std::endl;
    
    this->xData.push_back(MotionRepresentation::readXData(xNormalPath, xDataPath, sub_dir));

}

void
SingleControlWindow::
initCustomView()
{
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
            for(int i=0;i<2;i++)
            {
            	prevValues[i] = 0.0;
            	prevRewards[i] = 0.0;
            }
            recorder->recordCurrentFrame(mEnv);
            break;
        }
		case 'h':
			mEnv->reset();

			break;
		case 'g':
			mEnv->goBackEnvironment();

			break;
		case 'l':
			controlOn = !controlOn;
			break;
		case 'i':
			showCourtMesh = !showCourtMesh;
			break;
		case 'p':
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
			if(mPlay)
				step();
			else
			{
				if(windowFrame >= recorder->getNumFrames())
				{
					step();
					recorder->recordCurrentFrame(mEnv);
				}
				else
					recorder->loadFrame(mEnv, windowFrame++);
			}
			break;
		case '[':
			mFrame -= 100;
			if(mFrame <0)
				mFrame = 0;

			if(!mPlay)
				recorder->loadFrame(mEnv, windowFrame--);
			if(windowFrame < 0)
				windowFrame = 0;
			break;
		case ' ':
			if(!mPlay)
				windowFrame = recorder->loadLastFrame(mEnv);
			mPlay = !mPlay;
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
	{
		value = step();
		recorder->recordCurrentFrame(mEnv);
	}
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
}

int
SingleControlWindow::
step()
{
	std::cout<<"mEnv->curFrame : "<<mEnv->curFrame<<std::endl;
	std::chrono::time_point<std::chrono::system_clock> m_time_check_s = std::chrono::system_clock::now();
	if(mEnv->isFoulState())
	{
		sleep(1);
		std::cout<<"Foul Reset"<<std::endl;
		mEnv->foulReset();
	}

	if(mEnv->isTerminalState())
	{
		if(mTakeScreenShot)
		{
			for(int i=0;i<15;i++)
				screenshot();
		}
		else
		{
			sleep(1);
		}
		
		mEnv->slaveReset();
	}

	this->targetLocal.setZero();


	mEnv->getState(0);
	mEnv->saveEnvironment();
	if(mEnv->resetCount<=0)
		getActionFromNN(0);


	std::cout<<"action : "<<std::endl;
	std::cout<<mEnv->mActions[0].segment(0,5).transpose()<<std::endl;
	std::cout<<mEnv->mActions[0].segment(5,4).transpose()<<std::endl;
	std::cout<<mEnv->mActions[0].segment(9,6).transpose()<<std::endl;
	std::cout<<mEnv->mActions[0].segment(15,7).transpose()<<std::endl;
	std::cout<<std::endl;




	std::vector<std::vector<double>> concatControlVector;

	int resetDuration = mEnv->resetDuration;

	for(int id=0;id<1;++id)
	{
		if(mEnv->resetCount>resetDuration/2)
		{
			if(mEnv->randomPointTrajectoryStart)
			{
				mMotionGeneratorBatch->setBatchStateAndMotionGeneratorState(id, 
					mEnv->slaveResetPositionTrajectory[resetDuration - mEnv->resetCount], 
					mEnv->slaveResetBallPositionTrajectory[resetDuration - mEnv->resetCount]);
			}
			else
			{
				mMotionGeneratorBatch->setBatchStateAndMotionGeneratorState(id, mEnv->slaveResetPositionVector, mEnv->slaveResetBallPosition);
			}
		}
	}


	for(int id=0;id<1;++id)
	{
		if(mEnv->resetCount<=0)
		{
			concatControlVector.push_back(eigenToStdVec(mEnv->getMGAction(0)));
			mEnv->setAction(0, mActions[0]);
		}
		else
		{
			if(mEnv->resetCount>resetDuration)
			{
				concatControlVector.push_back(eigenToStdVec(mEnv->slaveResetTargetVector));
				Eigen::VectorXd actionTypeVector = mEnv->slaveResetTargetVector.segment(0,5);
				Eigen::VectorXd actionDetailVector(16);
				actionDetailVector = mEnv->slaveResetTargetVector.segment(5,16);
				mEnv->setActionType(0,getActionTypeFromVec(actionTypeVector));
				mEnv->setAction(0, actionDetailVector);

			}
			else
			{
				if(mEnv->randomPointTrajectoryStart)
				{
					concatControlVector.push_back(eigenToStdVec(mEnv->slaveResetTargetTrajectory[resetDuration-mEnv->resetCount]));
					Eigen::VectorXd actionTypeVector = mEnv->slaveResetTargetTrajectory[resetDuration-mEnv->resetCount].segment(0,5);
					Eigen::VectorXd actionDetailVector(16);
					actionDetailVector = mEnv->slaveResetTargetTrajectory[resetDuration-mEnv->resetCount].segment(5,16);
					// actionDetailVector.segment(4,5) = mEnv->slaveResetTargetTrajectory[resetDuration-mEnv->resetCount].segment(9,5);
					mEnv->setActionType(0,getActionTypeFromVec(actionTypeVector));
					mEnv->setAction(0, actionDetailVector);
				}
				else
				{
					concatControlVector.push_back(eigenToStdVec(mEnv->slaveResetTargetVector));
					Eigen::VectorXd actionTypeVector = mEnv->slaveResetTargetVector.segment(0,5);
					Eigen::VectorXd actionDetailVector(16);
					actionDetailVector = mEnv->slaveResetTargetVector.segment(5,16);
					// actionDetailVector.segment(4,5) = mEnv->slaveResetTargetVector.segment(9,5);
					mEnv->setActionType(0,getActionTypeFromVec(actionTypeVector));
					mEnv->setAction(0, actionDetailVector);
				}
			}

		}
	}

	std::vector<std::tuple<Eigen::VectorXd, Eigen::VectorXd, bool>>
	nextPoseAndContactsWithBatch = mMotionGeneratorBatch->generateNextPoseAndContactsWithBatch(concatControlVector);

	for(int id=0;id<1;++id)
	{
		mEnv->stepAtOnce(nextPoseAndContactsWithBatch[id]);
	}


    for(int i=0;i<2;i++)
    {
    	prevValues[i] = curValues[i];
    	prevRewards[i] = mEnv->curReward;
    }

	mEnv->getRewards();
	windowFrame++;
	mFrame++;

    std::chrono::duration<double> elapsed_seconds;
	elapsed_seconds = std::chrono::system_clock::now()-m_time_check_s;
    int calTime = std::min((int)(1000*elapsed_seconds.count()), 33);
	return calTime;
}


void
SingleControlWindow::
display()
{

	glClearColor(0.85, 0.85, 1.0, 1.0);
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

	GUI::drawSkeleton(mEnv->floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5), showCourtMesh, false);

	Eigen::Vector3d ballColor = Eigen::Vector3d(0.9, 0.8, 0.4);
	if(mEnv->curFrame < mEnv->throwingTime)
		ballColor = Eigen::Vector3d(0.9, 0.6, 0.0);
	GUI::drawSkeleton(mEnv->ballSkel, ballColor);

	Eigen::Vector3d skelColor(1.0, 1.0, 1.0);
	if(mEnv->resetCount >= 0)
		skelColor = Eigen::Vector3d(0.5, 0.5, 0.5);
	GUI::drawSkeleton(chars[0]->getSkeleton(), skelColor);

	Eigen::Isometry3d rootIsometry = mEnv->getRootT(0);

	glPushMatrix();
	Eigen::Vector3d rootPosition = rootIsometry.translation();
	glTranslated(rootPosition[0], rootPosition[1], rootPosition[2]);
	Eigen::AngleAxisd rootAA(rootIsometry.linear());
	glRotated(180/M_PI*rootAA.angle(), rootAA.axis()[0], rootAA.axis()[1], rootAA.axis()[2]);
	GUI::drawCoordinate(Eigen::Vector3d::Zero(), 0.2);

	glPopMatrix();


	// glPushMatrix();
	// glTranslated(mEnv->mTargetBallPosition[0], 0, mEnv->mTargetBallPosition[2]);
	// glBegin(GL_LINES);
 //    glColor3f(0.0, 0.0, 0.0);
 //    glVertex3f(0,0,0);
 //    glVertex3f(0,mEnv->mTargetBallPosition[1],0);
 //    glEnd();
 //    glPopMatrix();


	glLineWidth(1.0);
	glPushMatrix();
	glTranslated(mEnv->curBallPosition[0], 0, mEnv->curBallPosition[2]);
	glBegin(GL_LINES);
    glColor3f(0.0, 0.0, 0.0);
    glVertex3f(0,0,0);
    glVertex3f(0,mEnv->curBallPosition[1],0);
    glEnd();
	GUI::drawSphere(0.05, Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, 0.0));
    glPopMatrix();


    // Eigen::Vector3d targetBall2DPosition = mEnv->mTargetBallPosition;
    // targetBall2DPosition[1] = 0;


	for(int i=0;i<mEnv->mObstacles.size();i++)
	{
		glPushMatrix();
		glTranslated(mEnv->mObstacles[i][0], 1.0, mEnv->mObstacles[i][2]);
		GUI::drawCylinder(0.5, 2.0, Eigen::Vector3d(0.3, 0.3, 0.3));
		// glColor3f(0.0,0.0,1.0);
		// GUI::draw2dCircle(Eigen::Vector3d(0.0, 0.02, 0.0), Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitZ(),0.75, true);

		glPopMatrix();
	}

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
	    for(int i=0;i<numActions;i++)
	    {
	        if(xData[0][mFrame][i] >= 0.5)
	            curAction = std::to_string(i);
	    }
	    curAction = curAction+"     "+std::to_string(xData[0][mFrame][CRITICAL_OFFSET]/30.0);
    }
    else
    {

	    int maxIndex = 0;
	    double maxValue = -100;
	    for(int i=0;i<numActions;i++)
	    {
	        if(mEnv->mActions[0][i]> maxValue)
	        {
	        	maxValue= mEnv->mActions[0][i];
	        	maxIndex = i;
	        }
	    }
	    curAction = std::to_string(maxIndex);
	    curAction = curAction+"     "+std::to_string(mEnv->mActions[0][CRITICAL_OFFSET]/30.0);
    }

	Eigen::Vector3d ballTargetPosition =mEnv->mActions[0].segment(BALLTP_OFFSET,3)/100.0;
	Eigen::Vector3d ballTargetVelocity = mEnv->mActions[0].segment(BALLTV_OFFSET,3)/100.0;
	ballTargetPosition = rootIsometry * ballTargetPosition;
	ballTargetVelocity = rootIsometry.linear() * ballTargetVelocity;

	if(mEnv->mCurActionTypes[0] == 1 || mEnv->mCurActionTypes[0] == 3 )
	{
		GUI::drawArrow3D(ballTargetPosition, ballTargetVelocity, ballTargetVelocity.norm()/8.0, 0.05, Eigen::Vector3d(1.0, 0.0, 0.0), 0.08);
	}
	if(mEnv->mCurActionTypes[0] == 2)
	{
		GUI::drawSphere(0.15, ballTargetPosition, Eigen::Vector3d(1.0, 0.0, 0.0));
	}


    //Draw root to ball vector
    Eigen::Vector3d rootToBall;
    rootToBall[0] = 0.01*mEnv->mActions[0][2];
    rootToBall[2] = 0.01*mEnv->mActions[0][3];
    rootToBall[1] = 0.1;

    // std::cout<<"root to ball : "<<rootToBall.transpose()<<std::endl;
    glLineWidth(2.0);
   	GUI::drawLine(rootIsometry.translation(), rootIsometry*rootToBall, Eigen::Vector3d(1.0, 1.0, 1.0));
   	// GUI::drawLine(rootIsometry.translation(), rootIsometry*Eigen::Vector3d::UnitX(), Eigen::Vector3d(1.0, 0.0, 0.0));


    GUI::drawStringOnScreen(0.2, 0.25, std::to_string(mEnv->mActions[0][CRITICAL_OFFSET]/30.0), true, Eigen::Vector3d(1,1,1));

    std::string score = "Score : "+to_string(mEnv->mAccScore[0]);
   
	GUI::drawBoxOnScreen(0.25, 0.75, Eigen::Vector2d(15.0, 4.0),Eigen::Vector3d(1.0, 1.0, 1.0), true);
    GUI::drawStringOnScreen(0.2, 0.75, score, true, Eigen::Vector3d(0.0,0.0,0.0));


    GUI::drawStringOnScreen(0.2, 0.55, std::to_string(mEnv->mCurBallPossessions[0]), true, Eigen::Vector3d(1,1,1));

	showAvailableActions();

	GUI::drawBoxOnScreen(0.2+0.6*((double)mEnv->mCurActionTypes[0] / (numActions)), 0.2, Eigen::Vector2d(6.0, 4.0),Eigen::Vector3d(1.0, 1.0, 1.0));


    GUI::drawStringOnScreen(0.8, 0.55, std::to_string(curValues[0]), true, Eigen::Vector3d(1,1,1));
    GUI::drawStringOnScreen(0.8, 0.45, std::to_string(curValues[1]), true, Eigen::Vector3d(1,1,1));


    double tdError[2];

    for(int i=0;i<2;i++)
    {
 		tdError[i] = prevValues[i] - (prevRewards[i] + 0.999 * curValues[i]);
    }
    GUI::drawStringOnScreen(0.90, 0.55, std::to_string(tdError[0]), true, Eigen::Vector3d(1,1,1));
    GUI::drawStringOnScreen(0.90, 0.45, std::to_string(tdError[1]), true, Eigen::Vector3d(1,1,1));


	glutSwapBuffers();
	if(mTakeScreenShot)
	{
		screenshot();
	}
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
	int numActions = 5;

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

	std::cout<<"**ActionType : "<<mActionType.transpose()<<std::endl;
	std::cout<<"mEnv->curFrame : "<<mEnv->curFrame<<std::endl;
	std::cout<<"mEnv->resetCount : "<<mEnv->resetCount<<std::endl;

	if(mEnv->curFrame%mEnv->typeFreq == 0)
		mActionType = toOneHotVectorWithConstraint(index, mActionType);
	else
	{
		mActionType.setZero();
		int prevActionType = mEnv->mCurActionTypes[index];
		mActionType[prevActionType] = 1.0;

	}

	int actionType = getActionTypeFromVec(mActionType);

	Eigen::VectorXd mAction(mEnv->getNumAction() - numActions);

	Eigen::VectorXd state_1(state.size()+mActionType.rows());
	state_1.segment(0,state.size()) = state;
	state_1.segment(state.size(),mActionType.rows()) = mActionType;

	p::object get_action_1;

	get_action_1 = nn_module_1[index].attr("get_action_detail");

	p::tuple shape_1 = p::make_tuple(state_1.size());
	np::ndarray state_np_1 = np::empty(shape_1, dtype);

	float* dest_1 = reinterpret_cast<float*>(state_np_1.get_data());
	for(int j=0;j<state_1.size();j++)
	{
		dest_1[j] = state_1[j];
	}

	temp = get_action_1(state_np_1, actionType);
	np::ndarray action_np_1 = np::from_object(temp);
	float* srcs_1 = reinterpret_cast<float*>(action_np_1.get_data());

	for(int j=0;j<latentSize;j++)
	{
		mAction[j] = srcs_1[j];
	}

	Eigen::VectorXd encodedAction(latentSize);
	encodedAction = mAction.segment(0,encodedAction.size());
	Eigen::VectorXd decodedAction(16);

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

	for(int j=0;j<decodedAction.size();j++)
	{
		decodedAction[j] = src_d[j];
	}

	std::cout<<"Cur Action Type : "<<actionType<<std::endl;

	mActions[index] = mEnv->mNormalizer->denormalizeAction(decodedAction);
}

void
SingleControlWindow::
showAvailableActions()
{
	std::vector<int> aa = mEnv->mCharacters[0]->availableActionTypes;
	int numActionTypes = 5;
	for(int i=0;i<numActionTypes;i++)
	{
		if(std::find(aa.begin(), aa.end(), i) != aa.end())
			GUI::drawStringOnScreen_Big(-0.01 + 0.2+0.6*(((double) i)/numActionTypes), 0.2, mEnv->actionNameMap[i], Eigen::Vector3d::Ones());
		else
			GUI::drawStringOnScreen_Big(-0.01 + 0.2+0.6*(((double) i)/numActionTypes), 0.2, mEnv->actionNameMap[i], Eigen::Vector3d(0.5, 0.5, 0.5));

	}
}