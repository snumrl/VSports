#include "BvhWindow.h"
#include "../render/GLfunctionsDART.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include "../pyvs/EnvironmentPython.h"
#include <dart/dart.hpp>
#include <dart/utils/utils.hpp>
#include "../motion/BVHmanager.h"

// #include "./common/loadShader.h"
// #include <GL/glew.h>
#include <GL/glut.h>
// #include <GL/glew.h>
#include <GLFW/glfw3.h>
// Include GLM
#include <glm/glm.hpp>
#include "../extern/ICA/plugin/MotionGenerator.h"
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
BvhWindow::
initWindow(int _w, int _h, char* _name)
{
	mWindows.push_back(this);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE | GLUT_ACCUM);
	glutInitWindowPosition(500, 100);
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



BvhWindow::
BvhWindow()
:SimWindow(), mIsNNLoaded(false)
{
	mEnv = new Environment(30, 180, 4);
	mEnv->endTime = 300;
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
	// cout<<"444444"<<endl;
	controlOn = false;
	mActions.resize(mEnv->mNumChars);
	mActionNoises.resize(mEnv->mNumChars);
	for(int i=0;i<mActions.size();i++)
	{
		mActions[i] = Eigen::VectorXd(3);
		mActionNoises[i] = Eigen::VectorXd(3);
		mActions[i].setZero();
		mActionNoises[i].setZero();
	}
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

}

BvhWindow::
BvhWindow(const char* bvh_path, const char* nn_path)
:BvhWindow()
{
	targetActionType = 0;
	actionDelay = 0;
	bvhParser = new BVHparser(bvh_path, BVHType::BASKET);
	bvhParser->writeSkelFile();
	// cout<<bvhParser->skelFilePath<<endl;
	SkeletonPtr bvhSkel = dart::utils::SkelParser::readSkeleton(bvhParser->skelFilePath);
	charNames.push_back(getFileName_(bvh_path));
	// cout<<charNames[0]<<endl;	
	BVHmanager::setPositionFromBVH(bvhSkel, bvhParser, 0);
	mEnv->mWorld->addSkeleton(bvhSkel);

	// cout<<"Before MotionGenerator"<<endl;
	// exit(0);
	cout<<"BVH skeleton dofs : "<<bvhSkel->getNumDofs()<<endl;
	cout<<"BVH skeleton numBodies : "<<bvhSkel->getNumBodyNodes()<<endl;
	initDartNameIdMapping();
	mMotionGenerator = new ICA::dart::MotionGenerator(nn_path, this->dartNameIdMap);
	// cout<<bvhSkel->getPositions().transpose()<<endl;
	for(int i=0;i<10;i++)
	{
		BVHmanager::setPositionFromBVH(bvhSkel, bvhParser, 100+i);
		Eigen::VectorXd bvhPosition = bvhSkel->getPositions();
		// bvhPosition[3] -= 4.0;
		// cout<<bvhPosition.transpose()<<endl;
		mMotionGenerator->setCurrentPose(bvhPosition);
		bvhSkel->setPositions(bvhPosition);
	}

}

void
BvhWindow::
initDartNameIdMapping()
{    
	SkeletonPtr bvhSkel = mEnv->mWorld->getSkeleton(charNames[0]);
	int curIndex = 0;
	// cout<<bvhSkel->getNumBodyNodes()<<endl;
	for(int i=0;i<bvhSkel->getNumBodyNodes();i++)
	{
		this->dartNameIdMap[bvhSkel->getBodyNode(i)->getName()] = curIndex;
		curIndex += bvhSkel->getBodyNode(i)->getParentJoint()->getNumDofs();
	}

	// cout<<this->dartNameIdMap.size()<<endl;
	// for(auto& nameMap : this->dartNameIdMap)
	// {
	// 	cout<<nameMap.first<<" "<<nameMap.second<<endl;
	// }
}

void
BvhWindow::
initCustomView()
{
	// mCamera->eye = Eigen::Vector3d(3.60468, -4.29576, 1.87037);
	// mCamera->lookAt = Eigen::Vector3d(-0.0936473, 0.158113, 0.293854);
	// mCamera->up = Eigen::Vector3d(-0.132372, 0.231252, 0.963847);
	mCamera->eye = Eigen::Vector3d(-10.0, 5.0, 10.0);
	mCamera->lookAt = Eigen::Vector3d(0.0, 0.0, 0.0);
	mCamera->up = Eigen::Vector3d(0.0, 1.0, 0.0);

}

void
BvhWindow::
initGoalpost()
{
	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-8.0, 0.25 + floorDepth, 0.0), "red");
	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(8.0, 0.25 + floorDepth, 0.0), "blue");

	mWorld->addSkeleton(redGoalpostSkel);
	mWorld->addSkeleton(blueGoalpostSkel);
}



void
BvhWindow::
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

        case 'q': //dribble
        {
            targetActionType = 0;
            actionDelay = 0;
            break;
        }
        case 'e': //pass
        {
            targetActionType = 1;
            actionDelay = 30;
            break;
        }
        case 'r': //pass receive
        {
            targetActionType = 2;
            actionDelay = 30;
            break;
        }
        case 't': //shoot
        {
            targetActionType = 3;
            actionDelay = 30;
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
		case 'l':
			controlOn = !controlOn;
			break;
		case 'i':
			showCourtMesh = !showCourtMesh;
			break;

		default: SimWindow::keyboard(key, x, y);
	}
}
void
BvhWindow::
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
BvhWindow::
timer(int value)
{
	if(mPlay)
		step();
	// display();
	// glutSwapBuffers();
	glutPostRedisplay();
	SimWindow::timer(value);
}



void
BvhWindow::
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
BvhWindow::
applyMouseEvent()
{
    Eigen::Vector3d facing = mCamera->lookAt - mCamera->eye;
    this->targetLocal[2] = facing[0];
    this->targetLocal[3] = facing[2];
    this->targetLocal.segment(2,2).normalize();
    // std::cout<<targetLocal.segment(2,2).transpose()<<std::endl;
}

Eigen::Vector2d vec3dTo2d(Eigen::Vector3d vec3d)
{
	return Eigen::Vector2d(vec3d[0], vec3d[2]);
}

void
BvhWindow::
step()
{
	this->targetLocal.setZero();

	applyKeyBoardEvent();
	applyMouseEvent();
    actionDelay--;
    if(actionDelay < -30)
	{
		targetActionType = 0;
        actionDelay = 0;
	}

    // If terminal action goes to end, reset to dribble.
    if(targetActionType == 0)
        actionDelay = 0;


	SkeletonPtr bvhSkel = mEnv->mWorld->getSkeleton(charNames[0]);
	// for(int i=0;i<1;i++)
	// {
	// 	// BVHmanager::setPositionFromBVH(mEnv->mWorld->getSkeleton(charNames[i]), bvhParser, bvhFrame++);
	// }


	// for(int i=0;i<mEnv->mNumChars;i++)
	// {
	// 	mEnv->setAction(i, mActions[i]);



	// // }
	// this->targetLocal.segment(0,2) = this->goal - vec3dTo2d(bvhSkel->getPositions().segment(3,3));
	// this->targetLocal.segment(0,2) *= 100.0;


	// std::cout<<"Target Vel : "<<this->targetLocal.segment(0,2).transpose()<<std::endl;
	//Shoot direction is fixed to goalpost
	if(targetActionType == 3)
	{
		// this->targetLocal.segment(2,2) = Eigen::Vector2d(14.0-1.57,0.0)*0.8 - vec3dTo2d(bvhSkel->getPositions().segment(3,3));
		// this->targetLocal.segment(2,2).normalize();

		this->targetLocal.segment(4,3) = bvhSkel->getPositions().segment(3,3)+Eigen::Vector3d(0.0, 0.5, 0.0);
		this->targetLocal.segment(7,3) = Eigen::Vector3d(14.0-1.57, 10.0, 0.0)*0.8 - bvhSkel->getPositions().segment(3,3);
		this->targetLocal.segment(4,3) *= 100.0;
		this->targetLocal.segment(7,3) *= 20.0;
	}
	else if(targetActionType == 1)
	{
		this->targetLocal.segment(2,2) = this->targetLocal.segment(0,2).normalized();
	}


	std::cout<<"BVH skel position : "<<vec3dTo2d(bvhSkel->getPositions().segment(3,3)).transpose()<<endl;
	std::cout<<"View direction  : "<<vec3dTo2d(mCamera->lookAt - mCamera->eye).transpose().normalized()<<endl;
	
	// std::cout<<"BVH skel position : "<<vec3dTo2d(bvhSkel->getPositions().segment(3,3)).transpose()<<endl;

	Eigen::VectorXd nextPosition = mMotionGenerator->generateNextPose(this->targetLocal, targetActionType, actionDelay);
	bvhSkel->setPositions(nextPosition);
	// cout<<nextPosition.transpose()<<endl;

	// mEnv->stepAtOnce();
	// mEnv->getRewards();

}

void
BvhWindow::
display()
{
	glClearColor(0.85, 0.85, 1.0, 1.0);
	// glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	initLights();
	mCamera->apply();

	std::vector<Character2D*> chars = mEnv->getCharacters();


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


	GUI::drawSkeleton(mEnv->floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5), showCourtMesh, false);

	// GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.1, 0.1, 0.1));

	GUI::drawSkeleton(mEnv->mWorld->getSkeleton(charNames[0]));
	// cout<<"3333"<<endl;

	// std::string scoreString
	// = "Red : "+to_string((int)(mEnv->mAccScore[0] + mEnv->mAccScore[1]))+" |Blue : "+to_string((int)(mEnv->mAccScore[2]+mEnv->mAccScore[3]));

	std::string scoreString
	= "Red : "+to_string((mEnv->mAccScore[0]));//+" |Blue : "+to_string((int)(mEnv->mAccScore[1]));
	// = "Red : "+to_string((getRNDFeatureDiff(0)));//+" |Blue : "+to_string((int)(mEnv->mAccScore[1]));
	// cout<<"444444"<<endl;

	// cout<<mEnv->getCharacters()[0]->getSkeleton()->getVelocities().transpose()<<endl;
	// cout<<mActions[1][3]<<endl;

	GUI::drawStringOnScreen(0.2, 0.8, scoreString, true, Eigen::Vector3d::Zero());

	GUI::drawStringOnScreen(0.8, 0.8, to_string(mEnv->getElapsedTime()), true, Eigen::Vector3d::Zero());


	// GUI::drawVerticalLine(this->goal.segment(0,2), Eigen::Vector3d(1.0, 1.0, 1.0));

	glutSwapBuffers();
	if(mTakeScreenShot)
	{
		screenshot();
	}
	// glutPostRedisplay();
}

std::string
BvhWindow::
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
BvhWindow::
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
BvhWindow::
motion(int x, int y)
{
	SimWindow::motion(x, y);
}

