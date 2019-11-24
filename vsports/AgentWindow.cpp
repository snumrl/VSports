#include "AgentWindow.h"
#include "../render/GLfunctionsDART.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include "../pyvs/EnvironmentPython.h"
#include "./common/loadShader.h"
// #include <GL/glew.h>
#include <GL/glut.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>
// Include GLM
#include <glm/glm.hpp>
using namespace glm;
#include <iostream>
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;
extern enum key_state {NOTPUSHED, PUSHED} keyarr[127];

// std::chrono::time_point<std::chrono::system_clock> time_check_s = std::chrono::system_clock::now();

// void time_check_start()
// {
// 	time_check_s = std::chrono::system_clock::now();
// }

// void time_check_end()
// {
// 	std::chrono::duration<double> elapsed_seconds;
// 	elapsed_seconds = std::chrono::system_clock::now()-time_check_s;
// 	std::cout<<elapsed_seconds.count()<<std::endl;
// }

extern double floorDepth;

void
AgentWindow::
initWindow(int _w, int _h, char* _name)
{
	mWindows.push_back(this);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE | GLUT_ACCUM);
	glutInitWindowPosition(1500, 100);
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

	agentViewImg.resize(4*_w*_h);
	agentViewImgTemp.resize(4*_w*_h);

	// glfw.window_hint(glfw.VISIBLE, false);
	// glut.
}



AgentWindow::
AgentWindow(int index, Environment* env)
:SimWindow(), mIsNNLoaded(false), mIndex(index)
{
	mEnv = env;
	setAgentView();
	initGoalpost();

	mm = p::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = p::import("sys");

	boost::python::str module_dir = "../pyvs";
	sys_module.attr("path").attr("insert")(1, module_dir);
	// p::exec("import os",mns);
	// p::exec("import sys",mns);
	// p::exec("import math",mns);
	// p::exec("import sys",mns);
	p::exec("import torch",mns);
	p::exec("import torch.nn as nn",mns);
	p::exec("import torch.optim as optim",mns);
	p::exec("import torch.nn.functional as F",mns);
	p::exec("import torchvision.transforms as T",mns);
	p::exec("import numpy as np",mns);
	p::exec("from Model import *",mns);
	controlOn = false;
	mActions.resize(mEnv->mNumChars);
	for(int i=0;i<mActions.size();i++)
	{
		mActions[i] = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
	}
// GLenum err = glewInit();
// if (err != GLEW_OK)
//   cout<<"Not ok with glew"<<endl; // or handle the error in a nicer way
// if (!GLEW_VERSION_2_1)  // check that the machine supports the 2.1 API.
//   cout<<"Not ok with glew version"<<endl; // or handle the error in a nicer way
//   cout<< glewGetString(err) <<endl; // or handle the error in a nicer way

}

AgentWindow::
AgentWindow(int index, Environment* env, const std::string& nn_path)
:AgentWindow(index, env)
{
	mIsNNLoaded = true;


	p::str str = ("num_state = "+std::to_string(mEnv->getNumState())).c_str();
	p::exec(str,mns);
	str = ("num_action = "+std::to_string(mEnv->getNumAction())).c_str();
	p::exec(str, mns);

	nn_sc_module = new boost::python::object[mEnv->mNumChars];
	p::object *sc_load = new p::object[mEnv->mNumChars];
	reset_sc_hidden = new boost::python::object[mEnv->mNumChars];

	nn_la_module = new boost::python::object[mEnv->mNumChars];
	p::object *la_load = new p::object[mEnv->mNumChars];
	reset_la_hidden = new boost::python::object[mEnv->mNumChars];

	for(int i=0;i<mEnv->mNumChars;i++)
	{
		nn_sc_module[i] = p::eval("SchedulerNN(num_state, num_action).cuda()", mns);
		sc_load[i] = nn_sc_module[i].attr("load");
		reset_sc_hidden[i] = nn_sc_module[i].attr("reset_hidden");
		sc_load[i](nn_path+".pt");
		// if(i== 0|| i==1 || true)
		// 	sc_load[i](nn_path+".pt");
		// else
		// 	sc_load[i]("../save/goalReward/max.pt");
	}
	// cout<<"3344444444"<<endl;
	// mActions.resize(mEnv->mNumChars);
	// for(int i=0;i<mActions.size();i++)
	// {
	// 	mActions[i] = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
	// }
}

void
AgentWindow::
initCustomView()
{
	// mCamera->eye = Eigen::Vector3d(3.60468, -4.29576, 1.87037);
	// mCamera->lookAt = Eigen::Vector3d(-0.0936473, 0.158113, 0.293854);
	// mCamera->up = Eigen::Vector3d(-0.132372, 0.231252, 0.963847);
	mCamera->eye = Eigen::Vector3d(0.0, 0.0, 10.0);
	mCamera->lookAt = Eigen::Vector3d(0.0, 0.0, 0.0);
	mCamera->up = Eigen::Vector3d(0.0, 1.0, 0.0);

}

void
AgentWindow::
setAgentView()
{
	Eigen::Vector3d characterPosition;
	SkeletonPtr skel = mEnv->getCharacter(mIndex)->getSkeleton();
	characterPosition.segment(0,2) = skel->getPositions().segment(0,2);
	characterPosition[2] = 0.5;
	
	mCamera->lookAt = characterPosition;
	mCamera->up = Eigen::Vector3d(0.0, 0.0, 1.0);
	Eigen::AngleAxisd rotation = Eigen::AngleAxisd(skel->getPosition(2), mCamera->up);
	// cout<<mEnv->getCharacter(mIndex)->getDirection()<<endl;
	// mEnv->getCharacter(mIndex)->getSkeleton()->
	// setPosition(2, mEnv->getCharacter(mIndex)->getSkeleton()->getPosition(2)+0.02);

	mCamera->eye = characterPosition + rotation*Eigen::Vector3d(-1.2, 0.0, +0.2);

	// mCamera->fovy

}

void
AgentWindow::
initGoalpost()
{
	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-4.0, 0.0, 0.25 + floorDepth), "red");
	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(4.0, 0.0, 0.25 + floorDepth), "blue");

	mWorld->addSkeleton(redGoalpostSkel);
	mWorld->addSkeleton(blueGoalpostSkel);
}



void
AgentWindow::
keyboard(unsigned char key, int x, int y)
{
	SkeletonPtr manualSkel = mEnv->getCharacter(0)->getSkeleton();
	// cout<<"::::::::"<<endl;
	switch(key)
	{
		case 'c':
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
		case 'g':
			keyarr[int('g')] = PUSHED;
			break;
		case 'h':
			keyarr[int('h')] = PUSHED;
			break;
		case 'r':
			mEnv->reset();
			// for(int i=0;i<2;i++){
			// 	reset_sc_hidden[i]();
			// }


			// reset_hidden[2]();
			// reset_hidden[3]();
			break;
		case 'l':
			controlOn = !controlOn;
			break;

		default: SimWindow::keyboard(key, x, y);
	}
}
void
AgentWindow::
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
		// case 'h':
		// 	keyarr[int('h')] = NOTPUSHED;
		// 	break;
	}
}
void
AgentWindow::
timer(int value)
{
	// if(mPlay)
			step();


	// setAgentView();
	// display();
	SimWindow::timer(value);
	// glutTimerFunc(mDisplayTimeout, timerEvent, 1);
	// glutTimerFunc(mDisplayTimeout, timerEvent, 1);
	// glutPostRedisplay();
}

void
AgentWindow::
applyKeyEvent()
{
	double power = 1.0;
	SkeletonPtr skel = mEnv->getCharacter(mIndex)->getSkeleton();
	Eigen::AngleAxisd rotation = Eigen::AngleAxisd(skel->getPosition(2), mCamera->up);
	Eigen::Vector2d viewDirection = (rotation* Eigen::Vector3d::UnitX()).segment(0,2);
	Eigen::Vector2d rightDirection = (rotation*Eigen::Vector3d::UnitY()).segment(0,2);
	if(keyarr[int('w')]==PUSHED)
	{
		// mActions[0][1] += power;
		mActions[0].segment(0,2) += viewDirection * power;
		// cout<<mActions[0].segment(0,2).transpose()<<endl;

		// mActions[].segment(0,2) = Eigen::Vector3d(1.0, 0.0, 0.0)
	}
	if(keyarr[int('s')]==PUSHED)
	{
		mActions[0].segment(0,2) += -viewDirection * power;
		// mActions[0][1] += -power;
	}
	if(keyarr[int('a')]==PUSHED)
	{
		mActions[0].segment(0,2) += rightDirection * power;
		// mActions[0][0] += -power;
	}
	if(keyarr[int('d')]==PUSHED)
	{
		mActions[0].segment(0,2) += -rightDirection * power;
		// mActions[0][0] += power;
	}
	if(keyarr[int('g')]==PUSHED)
	{
		mActions[0][3] = 1.0;
	}
	else
	{
		// mActions[0][3] -= 0.1;
		// if(mActions[0][3] < -0.1)
		// 	mActions[0][3] = -0.1;
		mActions[0][3] = 0.0;
	}
	// cout<<"!!!!!!"<<endl;

	skel->setVelocity(0, mEnv->maxVel*mActions[0][0]);
	skel->setVelocity(1, mEnv->maxVel*mActions[0][1]);
	mActions[0].segment(0,2).setZero();
	// if(keyarr[int('h')]==PUSHED)
	// {
	// 	mActions[0][3] = 1.0;
	// 	Eigen::Vector3d curVel = mEnv->getState(0);
	// }
}

void
AgentWindow::
step()
{
	if(mEnv->isTerminalState())
	{
		sleep(1);
		mEnv->reset();
	}

	if(mIsNNLoaded)
		getActionFromNN(true);

	// cout<<"step in agent"<<endl;

	if(mEnv->mNumChars == 4)
	{
		// mActions[0] = mEnv->getActionFromBTree(0);
		mActions[1] = mEnv->getActionFromBTree(1);
		mActions[2] = mEnv->getActionFromBTree(2);
		mActions[3] = mEnv->getActionFromBTree(3);
	}
	else if(mEnv->mNumChars == 2)
	{
		// mActions[0] = mEnv->getActionFromBTree(0);
		// mActions[1] = mEnv->getActionFromBTree(1);
	}
	// mActions[0] = mEnv->getActionFromBTree(0);
	// mActions[1] = mEnv->getActionFromBTree(1);
	// mActions[2] = mEnv->getActionFromBTree(2);
	// mActions[3] = mEnv->getActionFromBTree(3);
	// cout<<mActions[1].transpose()<<endl;
	applyKeyEvent();

	// cout<<mEnv->getCharacter(0)->getSkeleton()->getVelocities().transpose()<<endl;
	for(int i=0;i<mEnv->mNumChars;i++)
	{
		mEnv->getState(i);
		// mActions[i] = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
		mEnv->setAction(i, mActions[i]);
	}

	mEnv->stepAtOnce();
	mEnv->getRewards();
	for(int i=0;i<mActions.size();i++)
	{
		mActions[i].segment(0,3) = Eigen::Vector3d(0.0, 0.0, 0.0);
	}
}

void
AgentWindow::
getActionFromNN(bool vsHardcodedAI)
{
	p::object get_sc_action;
	p::object get_la_action;

	mActions.clear();
	mActions.resize(2);

	for(int i=0;i<mEnv->mNumChars;i++)
	{
		Eigen::VectorXd state = mEnv->getState(i);

		Eigen::VectorXd mAction(mEnv->getNumAction());
		if((vsHardcodedAI && (i == 1)))
		{

			for(int j=0;j<3;j++)
			{
				mAction[j] = 0.0;
			}

			mAction[3] = 1.0;
			mAction[3] = -1.0;
			mActions[i] = mAction;
		}
		else
		{
			get_sc_action = nn_sc_module[i].attr("get_action");

			p::tuple shape = p::make_tuple(state.size());
			np::dtype dtype = np::dtype::get_builtin<float>();
			np::ndarray state_np = np::empty(shape, dtype);

			float* dest = reinterpret_cast<float*>(state_np.get_data());
			for(int j=0;j<state.size();j++)
			{
				dest[j] = state[j];
			}

			p::object temp = get_sc_action(state_np);
			np::ndarray action_np = np::from_object(temp);
			float* srcs = reinterpret_cast<float*>(action_np.get_data());
			for(int j=0;j<mAction.rows();j++)
			{
				mAction[j] = srcs[j];
			}
			// if(i==0)
			// 	cout<<i<<" "<<mAction[2]<<endl;	
			mActions[i] = mAction;



		}
	}
}

Eigen::VectorXd
AgentWindow::
getValueGradient(int index)
{
	p::object get_value_gradient;
	get_value_gradient = nn_sc_module[0].attr("get_value_gradient");

	Eigen::VectorXd state = mEnv->getState(index);
	Eigen::VectorXd valueGradient(state.size());

	p::tuple shape = p::make_tuple(state.size());
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape, dtype);
	float* dest = reinterpret_cast<float*>(state_np.get_data());
	for(int j=0;j<state.size();j++)
	{
		dest[j] = state[j];
	}

	p::object temp = get_value_gradient(state_np);
	np::ndarray valueGradient_np = np::from_object(temp);
	float* srcs = reinterpret_cast<float*>(valueGradient_np.get_data());
	for(int j=0;j<valueGradient.rows();j++)
	{
		valueGradient[j] = srcs[j];
	}
	return valueGradient;
}

void
AgentWindow::
display()
{
	glClearColor(0.85, 0.85, 1.0, 1.0);
	// glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	initLights();
	setAgentView();
	mCamera->apply();

	std::vector<Character2D*> chars = mEnv->getCharacters();


	for(int i=0;i<chars.size();i++)
	{
		// if (i!=0)
		// 	continue;
		if(chars[i]->getTeamName() == "A")
			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
		// else
		// 	GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));

	}

	GUI::drawSkeleton(mEnv->floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5));

	GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.1, 0.1, 0.1));


	GUI::drawSkeleton(mEnv->wallSkel, Eigen::Vector3d(0.5,0.5,0.5));

	// Not simulated just for see

	GUI::drawSkeleton(redGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0), true);

	GUI::drawSkeleton(blueGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));
	// cout<<"3333"<<endl;

	// std::string scoreString
	// = "Red : "+to_string((int)(mEnv->mAccScore[0] + mEnv->mAccScore[1]))+" |Blue : "+to_string((int)(mEnv->mAccScore[2]+mEnv->mAccScore[3]));

	// std::string scoreString
	// = "Red : "+to_string((int)(mEnv->mAccScore[0]));

	//+" |Blue : "+to_string((int)(mEnv->mAccScore[1]));
	// cout<<"444444"<<endl;


	// GUI::drawStringOnScreen(0.2, 0.8, scoreString, true, Eigen::Vector3d::Zero());

	// GUI::drawStringOnScreen(0.8, 0.8, to_string(mEnv->getElapsedTime()), true, Eigen::Vector3d::Zero());

	// cout<<"5555555"<<endl;

	


	GUI::drawSoccerLine(8, 6);

	glutSwapBuffers();
	if(mTakeScreenShot)
	{
		screenshot();
	}
	glutPostRedisplay();

	// getAgentView();
	// screenshot();
}

std::string
AgentWindow::
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
AgentWindow::
drawValueGradient()
{
	int numStates = mEnv->mStates[0].size();
	// GUI::drawStringOnScreen(0.8, 0.8, to_string(mEnv->getElapsedTime()), true, Eigen::Vector3d::Zero());

	// double leftOffset = 0.02;
	// double rightOffset = 0.04;



	glPushMatrix();
	double boxSize = 1.0 / (numStates-1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0,0.0,1.0,
			0.0,0.0, 0.0,
			0.0, 0.0 + 1.0,0.0);

	glTranslated(-0.5, -0.5, 0.0);

	GUI::drawValueGradientBox(mEnv->mStates[0], getValueGradient(0), boxSize);


	glPopMatrix();

	GLint w = glutGet(GLUT_WINDOW_WIDTH);
	GLint h = glutGet(GLUT_WINDOW_HEIGHT);

	for(int i=0;i<numStates;i++)
	{
		Eigen::Vector3d eyeToBox = Eigen::Vector3d(i * boxSize - 0.5, -0.5, 0.0);
		double fovx = mCamera->fovy * w / h;

		double boxAngleX = atan((eyeToBox[0] - boxSize/3.0)/(1.0 + boxSize)) / M_PI * 180.0;
		double boxAngleY = atan((eyeToBox[1]+boxSize)/(1.0 + boxSize)) / M_PI * 180.0; 

		// cout<<i<<endl;

		GUI::drawStringOnScreen_small(0.5 + boxAngleX/fovx, 0.5  + boxAngleY/mCamera->fovy, indexToStateString(i), Eigen::Vector3d::Zero());

	}


}


void
AgentWindow::
mouse(int button, int state, int x, int y) 
{
	SimWindow::mouse(button, state, x, y);
}


void
AgentWindow::
motion(int x, int y)
{

	// if (!mIsDrag)
	// 	return;

	// int mod = glutGetModifiers();
	// if(mMouseType == GLUT_LEFT_BUTTON)
	// {
	// 	mCamera->rotate(x, y, mPrevX, mPrevY);
	// }
	// else if(mMouseType == GLUT_RIGHT_BUTTON)
	// {
	// 	mCamera->translate(x, y, mPrevX, mPrevY);
	// }
	SkeletonPtr skel = mEnv->getCharacter(mIndex)->getSkeleton();
	skel->setPosition(2, skel->getPosition(2) + (mPrevX - x)/100.0);


	mPrevX = x;
	mPrevY = y;

	// SimWindow::motion(x, y);
}

void
AgentWindow::
getAgentView()
{
	int tw = glutGet(GLUT_WINDOW_WIDTH);
	int th = glutGet(GLUT_WINDOW_HEIGHT);

	glReadPixels(0, 0,  tw, th, GL_RGBA, GL_UNSIGNED_BYTE, &agentViewImgTemp[0]);

	// reverse temp2 temp1
	for (int row = 0; row < th; row++) {
		memcpy(&agentViewImg[row * tw * 4],
		&agentViewImgTemp[(th - row - 1) * tw * 4], tw * 4);
	}

}