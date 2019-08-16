#include "SimpleSocWindow.h"
#include "../render/GLfunctionsDART.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include "../pyvs/EnvironmentPython.h"
#include <GL/glut.h>
#include <iostream>
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

std::chrono::time_point<std::chrono::system_clock> time_check_s = std::chrono::system_clock::now();

void time_check_start()
{
	time_check_s = std::chrono::system_clock::now();
}

void time_check_end()
{
	std::chrono::duration<double> elapsed_seconds;
	elapsed_seconds = std::chrono::system_clock::now()-time_check_s;
	std::cout<<elapsed_seconds.count()<<std::endl;
}

double floorDepth = -0.1;

SimpleSocWindow::
SimpleSocWindow()
:SimWindow(),vsHardcodedAI_difficulty(3.0)
{
	mEnv = new Environment(30, 600, 4);
	initCustomView();
	initGoalpost();
	mActions = mEnv->mActions;



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
}

SimpleSocWindow::
SimpleSocWindow(const std::string& nn_path)
:SimpleSocWindow()
{
	mIsNNLoaded = true;


	p::str str = ("num_state = "+std::to_string(mEnv->getNumState())).c_str();
	p::exec(str,mns);
	str = ("num_action = "+std::to_string(mEnv->getNumAction())).c_str();
	p::exec(str, mns);
	// str = "use_cuda = torch.cuda.is_available()";
	// p::exec(str, mns);


	nn_module = p::eval("NoCNNSimulationNN(num_state, num_action).cuda()", mns);

	p::object load = nn_module.attr("load");
	reset_hidden = nn_module.attr("reset_hidden"); 
	load(nn_path);
}

void
SimpleSocWindow::
initCustomView()
{
	mCamera->eye = Eigen::Vector3d(3.60468, -4.29576, 1.87037);
	mCamera->lookAt = Eigen::Vector3d(-0.0936473, 0.158113, 0.293854);
	mCamera->up = Eigen::Vector3d(-0.132372, 0.231252, 0.963847);
}

void
SimpleSocWindow::
initGoalpost()
{
	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-4.0, 0.0, 0.25 + floorDepth), "red");
	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(4.0, 0.0, 0.25 + floorDepth), "blue");

	mWorld->addSkeleton(redGoalpostSkel);
	mWorld->addSkeleton(blueGoalpostSkel);
}



void
SimpleSocWindow::
keyboard(unsigned char key, int x, int y)
{
	bool controlOn = false;
	SkeletonPtr manualSkel = mEnv->getCharacter(0)->getSkeleton();

	switch(key)
	{
		// case 'c':
		// 	cout<<mCamera->eye.transpose()<<endl;
		// 	cout<<mCamera->lookAt.transpose()<<endl;
		// 	cout<<mCamera->up.transpose()<<endl;
		// 	break;
		case 'w':
			if(controlOn)
				manualSkel->setVelocities(Eigen::Vector2d(-3.0, 0.0));
			break;
		case 's':
			if(controlOn)
				manualSkel->setVelocities(Eigen::Vector2d(3.0, 0.0));
			break;
		case 'a':
			if(controlOn)
				manualSkel->setVelocities(Eigen::Vector2d(0.0, -3.0));
			break;
		case 'd':
			if(controlOn)
				manualSkel->setVelocities(Eigen::Vector2d(0.0, 3.0));
			break;
		case 'r':
			mEnv->reset();
			reset_hidden();
			break;
		case ']':
			vsHardcodedAI_difficulty += 0.1;

			if(vsHardcodedAI_difficulty>5.0)
				vsHardcodedAI_difficulty = 5.0;
			cout<<vsHardcodedAI_difficulty<<endl;
			break;
		case '[':
			vsHardcodedAI_difficulty += -0.1;
			if(vsHardcodedAI_difficulty<0.0)
				vsHardcodedAI_difficulty = 0.0;
			cout<<vsHardcodedAI_difficulty<<endl;
			break;

		default: SimWindow::keyboard(key, x, y);
	}
}

void
SimpleSocWindow::
timer(int value)
{
	if(mPlay)
		step();
	SimWindow::timer(value);
}

void
SimpleSocWindow::
step()
{
	// cout<<"????????"<<endl;
	getActionFromNN(true, true);
	// std::cout<<"step!"<<std::endl;
	for(int i=0;i<mEnv->mNumChars;i++)
	{
		// cout<<mActions[i].transpose()<<endl;
		// dart::collision::CollisionDetectorPtr detector = mEnv->mWorld->getConstraintSolver()->getCollisionDetector();
		// auto wall_char_collisionGroup = detector->createCollisionGroup(mEnv->wallSkel->getBodyNodes(), 
		// 	mEnv->getCharacter(i)->getSkeleton()->getRootBodyNode());
		// bool collision = wall_char_collisionGroup->collide();
		// if(collision)
		// {
		// 	mEnv->setAction(i, Eigen::VectorXd::Zero(mActions[i].size()));
		// 	cout<<"collide!"<<endl;
		// }
		// else
		mEnv->setAction(i, mActions[i]);
	}

	mEnv->mNumIterations = 400;
	int sim_per_control = mEnv->getSimulationHz()/mEnv->getControlHz();
	for(int i=0;i<sim_per_control;i++)
	{
		mEnv->step();
	}

}

void
SimpleSocWindow::
getActionFromNN(bool vsHardcodedAI, bool isRNN)
{
	// cout<<"getActionFromNN"<<endl;
	p::object get_action;
	mActions.clear();
	if(!isRNN)
		get_action = nn_module.attr("get_action");
	else
		get_action = nn_module.attr("get_action_rnn");
	for(int i=0;i<mEnv->mNumChars;i++)
	{

		Eigen::VectorXd mAction(mEnv->getNumAction());
		std::vector<double> state = mEnv->getState(i);
		// mEnv->getState(i);
		// Eigen::VectorXd state = mEnv->mStates[i];
		//change the i=1 agent
		if(vsHardcodedAI && (i==2 || i == 3))
		{
			// cout<<"i : "<<i<<endl;
			Eigen::VectorXd curBallRelaltionalP = mEnv->mStates[i].segment(ID_BALL_P,2);
			Eigen::VectorXd direction = curBallRelaltionalP.normalized();
			Eigen::VectorXd curVel = mEnv->mStates[i].segment(ID_V,2);
			mAction.segment(0, 2) = (direction*vsHardcodedAI_difficulty - curVel);
			mAction[2] = rand()%2;
			// mAction[2] = 0;
			mActions.push_back(mAction);
		}
		else
		{
			// cout<<"i : "<<i<<endl;
			p::tuple shape = p::make_tuple(state.size());
			np::dtype dtype = np::dtype::get_builtin<float>();
			np::ndarray state_np = np::empty(shape, dtype);

			// cout<<state.segment(0,6).transpose()<<endl;
			// cout<<shape<<endl;
			// cout<<"11111"<<endl;
			float* dest = reinterpret_cast<float*>(state_np.get_data());
			// cout<<"22222"<<endl;
			for(int j=0;j<state.size();j++)
			{
				dest[j] = state[j];
			}

			// cout<<"33333"<<endl;
	// time_check_start();

			p::object temp = get_action(state_np);
			np::ndarray action_np = np::from_object(temp);
			float* srcs = reinterpret_cast<float*>(action_np.get_data());
			for(int j=0;j<mAction.rows();j++)
			{
				mAction[j] = srcs[j];
			}
			mActions.push_back(mAction);

		}
	}
	// cout<<endl;
}


void
SimpleSocWindow::
display()
{
	glClearColor(0.85, 0.85, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	initLights();
	mCamera->apply();

	std::vector<Character2D*> chars = mEnv->getCharacters();

	// exit(0);
	// mEnv->getState_map(0);
	// exit(0);

	for(int i=0;i<chars.size();i++)
	{
		if(chars[i]->getTeamName() == "A")
			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
		else
			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));


	}
	// GUI::drawSkeleton(chars[0]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
	// GUI::drawSkeleton(chars[1]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));


	// for(int i=0;i<2;i++)
	// {
	// 	GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
	// }
	// for(int i=2;i<4;i++)
	// {
	// 	GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));
	// }

	GUI::drawSkeleton(mEnv->floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5));
	if(mEnv->mScoreBoard[0] == 1)
		GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.7, 0.1, 0.1));
	else if(mEnv->mScoreBoard[0] == 0)
		GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.1, 0.1, 0.7));
	else
		GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.1, 0.1, 0.1));

	GUI::drawSkeleton(mEnv->wallSkel, Eigen::Vector3d(0.5,0.5,0.5));

	// Not simulated just for see
	GUI::drawSkeleton(redGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));
	GUI::drawSkeleton(blueGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));

	std::string scoreString
	= "Red : "+to_string((int)mEnv->mAccScore[0])+" |Blue : "+to_string((int)mEnv->mAccScore[2]);



	GUI::drawStringOnScreen(0.2, 0.8, scoreString, true, Eigen::Vector3d::Zero());


	glutSwapBuffers();
	if(mTakeScreenShot)
	{
		screenshot();
	}
	glutPostRedisplay();
}

void
SimpleSocWindow::
mouse(int button, int state, int x, int y) 
{
	SimWindow::mouse(button, state, x, y);
}


void
SimpleSocWindow::
motion(int x, int y)
{
	SimWindow::motion(x, y);
}

