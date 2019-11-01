#include "SLACWindow.h"
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

SLACWindow::
SLACWindow()
:SimWindow(),vsHardcodedAI_difficulty(4.0)
{
	mEnv = new Environment(30, 600, 4);
	initCustomView();
	initGoalpost();

	mSubgoalCharacters.resize(1);
	mSubgoalCharacters[0] = new Character2D("A_0_subgoal");

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
}

SLACWindow::
SLACWindow(const std::string& nn_path)
:SLACWindow()
{
	mIsNNLoaded = true;


	p::str str = ("num_state = "+std::to_string(mEnv->getNumState())).c_str();
	p::exec(str,mns);
	str = ("num_action = "+std::to_string(mEnv->getNumAction())).c_str();
	p::exec(str, mns);
	// str = "use_cuda = torch.cuda.is_available()";
	// p::exec(str, mns);


	// nn_module[0] = p::eval("NoCNNSimulationNN(num_state, num_action).cuda()", mns);
	// nn_module[1] = p::eval("NoCNNSimulationNN(num_state, num_action).cuda()", mns);

	// load[0] = nn_module[0].attr("load");
	// load[1] = nn_module[1].attr("load");

	// reset_hidden[0] = nn_module[0].attr("reset_hidden"); 
	// reset_hidden[1] = nn_module[1].attr("reset_hidden"); 
	// load[0](nn_path);
	// load[1](nn_path);
	nn_sc_module = new boost::python::object[mEnv->mNumChars];
	p::object *sc_load = new p::object[mEnv->mNumChars];
	reset_sc_hidden = new boost::python::object[mEnv->mNumChars];

	nn_la_module = new boost::python::object[mEnv->mNumChars];
	p::object *la_load = new p::object[mEnv->mNumChars];
	reset_la_hidden = new boost::python::object[mEnv->mNumChars];

	// cout<<"33333333333"<<endl;
	for(int i=0;i<mEnv->mNumChars;i++)
	{
		nn_sc_module[i] = p::eval("SchedulerNN(num_state, num_action).cuda()", mns);
		sc_load[i] = nn_sc_module[i].attr("load");
		reset_sc_hidden[i] = nn_sc_module[i].attr("reset_hidden");
		if(i== 0|| i==1 || true)
			sc_load[i](nn_path+"_sc.pt");
		else
			sc_load[i]("../save/goalReward/max_sc.pt");


		nn_la_module[i] = p::eval("LActorNN(num_state, num_action).cuda()", mns);
		la_load[i] = nn_la_module[i].attr("load");
		reset_la_hidden[i] = nn_la_module[i].attr("reset_hidden");
		la_load[i](nn_path+"_la.pt");
	}
	// cout<<"3344444444"<<endl;
	mActions.resize(mEnv->mNumChars);
	mSubgoalStates.resize(mEnv->mNumChars);
	mSubgoalStates[0].resize(mEnv->getNumState());
	mSubgoalStates[0].setZero();
}

void
SLACWindow::
initCustomView()
{
	mCamera->eye = Eigen::Vector3d(3.60468, -4.29576, 1.87037);
	mCamera->lookAt = Eigen::Vector3d(-0.0936473, 0.158113, 0.293854);
	mCamera->up = Eigen::Vector3d(-0.132372, 0.231252, 0.963847);
}

void
SLACWindow::
initGoalpost()
{
	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-4.0, 0.0, 0.25 + floorDepth), "red");
	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(4.0, 0.0, 0.25 + floorDepth), "blue");

	mWorld->addSkeleton(redGoalpostSkel);
	mWorld->addSkeleton(blueGoalpostSkel);
}



void
SLACWindow::
keyboard(unsigned char key, int x, int y)
{
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
			for(int i=0;i<2;i++){
				reset_sc_hidden[i]();
				reset_la_hidden[i]();
			}


			// reset_hidden[2]();
			// reset_hidden[3]();
			break;
		case 'l':
			controlOn = !controlOn;
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
SLACWindow::
timer(int value)
{
	if(mPlay)
		step();
	SimWindow::timer(value);
}

void
SLACWindow::
step()
{
	if(mEnv->isTerminalState())
	{
		// for(int i=0;i<4;i++)
		// {
		// 	for(int j=mEnv->getNumState()-8;j<mEnv->getNumState();j++)
		// 	{
		// 		cout<<mEnv->getState(i)[j]<<" ";
		// 	}
		// 	cout<<endl;
		// }

		sleep(1);
		mEnv->reset();
	}
	// cout<<"????????"<<endl;
	getSubgoalFromSchedulerNN(true);
	// getActionFromLActorNN(true);
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
		// if( i== 0)
		if(!controlOn)
			mEnv->setAction(i, mActions[i]);
		else
			mEnv->setAction(i, Eigen::Vector3d(0.0, 0.0, -1.0));
		// cout<<i<<" "<<mActions[i][2]<<endl;
	}

	mEnv->mNumIterations = 100;
	// int sim_per_control = mEnv->getSimulationHz()/mEnv->getControlHz();

	mEnv->stepAtOnce();
	// cout<<mEnv->getSchedulerReward(0)<<endl;
	mEnv->getRewards();
	// for(int i=0;i<sim_per_control;i++)
	// {
	// 	mEnv->step();
	// }

}

void
SLACWindow::
getSubgoalFromSchedulerNN(bool vsHardcodedAI)
{
	p::object get_sc_action;
	p::object get_la_action;

	mActions.clear();
	mActions.resize(2);
	// mSubgoalStates.clear();
	// mSubgoalStates.resize(2);
	// mWSubgoalStates.clear();
	// mWSubgoalStates.resize(2);

	for(int i=0;i<mEnv->mNumChars;i++)
	{
		// if(i!=0)
		// 	continue;
		// cout<<"############"<<endl;
		Eigen::VectorXd mAction(mEnv->getNumAction());
		// Eigen::VectorXd mSubgoalState(mEnv->getNumState());
		// Eigen::VectorXd mWSubgoalState(mEnv->getNumState());

		Eigen::VectorXd state = mEnv->getSchedulerState(i);
		if(i==0)
		{
		// cout<<state.transpose()<<endl;
			// mEnv->getCharacter(0)->getSkeleton()->setPositions(4.0 *(state.segment(0,2) + state.segment(13,2)));
			// cout<<state.segment(13,2).transpose()<<endl;
		}
		// mEnv->getState(i);
		// Eigen::VectorXd state = mEnv->mSimpleStates[i];
		//change the i=1 agent
		if((vsHardcodedAI && (i == 1)) && false)
		{
				// cout<<"0000"<<endl;
			// cout<<"i : "<<i<<endl;
			// Eigen::VectorXd curBallRelaltionalP = mEnv->mSimpleStates[i].segment(ID_BALL_P,2);
			// Eigen::VectorXd direction = curBallRelaltionalP.normalized();
			// Eigen::VectorXd curVel = mEnv->mSimpleStates[i].segment(ID_V,2);
			// mAction.segment(0, 2) = (direction*vsHardcodedAI_difficulty - curVel);

			for(int j=0;j<2;j++)
			{
				// if(mAction[j] > 0.5)
				// 	mAction[j] = 0.5;
				// else if(mAction[j] < -0.5)
				// 	mAction[j] = -0.5;

				mAction[j] = 0.0;
			}
			// cout<<mAction.size()<<endl;

			// mAction[2] = rand()%3-1;
			mAction[2] = 1.0;
			mAction[2] = -1.0;
			mActions[i] = mAction;
		}
		else
		{
			// cout<<"111111"<<endl;
			get_sc_action = nn_sc_module[i].attr("get_action");
			// get_la_action = nn_la_module[i].attr("get_action");
			// cout<<"i : "<<i<<endl;
			p::tuple shape = p::make_tuple(state.size());
			np::dtype dtype = np::dtype::get_builtin<float>();
			np::ndarray state_np = np::empty(shape, dtype);
			// cout<<"22222"<<endl;

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

			// p::object temp = get_sc_action(state_np);
			// np::ndarray action_np = np::from_object(temp);
			// float* srcs = reinterpret_cast<float*>(action_np.get_data());
			// for(int j=0;j<mSubgoalState.rows();j++)
			// {
			// 	mSubgoalState[j] = srcs[j];
			// 	mWSubgoalState[j] = srcs[j+mSubgoalState.rows()];
			// }

			// Eigen::VectorXd linearActorState(mSubgoalState.size()*2);
			// linearActorState.segment(0, mSubgoalState.size()) = mSubgoalState;
			// linearActorState.segment(mSubgoalState.size(), mSubgoalState.size()) = mWSubgoalState;

			// mEnv->setLinearActorState(0, linearActorState);

			// mSubgoalStates[i] = mEnv->unNormalizeNNState(mSubgoalState);
			// cout<<"@@@@@@"<<endl;
			p::object temp = get_sc_action(state_np);
			np::ndarray action_np = np::from_object(temp);
			float* srcs = reinterpret_cast<float*>(action_np.get_data());
			for(int j=0;j<mAction.rows();j++)
			{
				mAction[j] = srcs[j];
			}
			if(i==0)
				cout<<i<<" "<<mAction[2]<<endl;	
			mActions[i] = mAction;



		}
	}
}

Eigen::VectorXd
SLACWindow::
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
SLACWindow::
getActionFromLActorNN(bool vsHardcodedAI)
{
	// cout<<"getActionFromNN"<<endl;
	p::object get_sc_action;
	p::object get_la_action;
	mActions.clear();
	mActions.resize(2);
	
	for(int i=0;i<mEnv->mNumChars;i++)
	{

		Eigen::VectorXd mAction(mEnv->getNumAction());
		Eigen::VectorXd state = mEnv->getState(i);

		Eigen::VectorXd lactorState(state.size()*2);

		// mEnv->getState(i);
		// Eigen::VectorXd state = mEnv->mSimpleStates[i];
		//change the i=1 agent
		if(vsHardcodedAI && (i == 1) && false)
		{
			// cout<<"i : "<<i<<endl;
			// Eigen::VectorXd curBallRelaltionalP = mEnv->mSimpleStates[i].segment(ID_BALL_P,2);
			// Eigen::VectorXd direction = curBallRelaltionalP.normalized();
			// Eigen::VectorXd curVel = mEnv->mSimpleStates[i].segment(ID_V,2);
			// mAction.segment(0, 2) = (direction*vsHardcodedAI_difficulty - curVel);

			for(int j=0;j<2;j++)
			{
				// if(mAction[j] > 0.5)
				// 	mAction[j] = 0.5;
				// else if(mAction[j] < -0.5)
				// 	mAction[j] = -0.5;

				mAction[j] = 0.0;
			}

			// mAction[2] = rand()%3-1;
			// mAction[2] = 1.0;
			mAction[2] = 0;
			mActions[i] = mAction;
		}
		else
		{
			// lactorState.segment(0, state.size()) = state;
			// lactorState.segment(state.size(), state.size()) = mEnv->normalizeNNState(mSubgoalStates[i]);
			lactorState = mEnv->getLinearActorState(0);
			// cout<<lactorState.segment(state.size(), state.size()).segment(_ID_BALL_P, 2).transpose()<<endl;
			// cout<<mSubgoalStates[i].segment(_ID_BALL_P, 2).transpose()<<endl;
			// get_sc_action = nn_sc_module[i].attr("get_action");
			get_la_action = nn_la_module[i].attr("get_action");
			// cout<<"i : "<<i<<endl;
			p::tuple shape = p::make_tuple(lactorState.size());
			np::dtype dtype = np::dtype::get_builtin<float>();
			np::ndarray state_np = np::empty(shape, dtype);

			// cout<<state.segment(0,6).transpose()<<endl;
			// cout<<shape<<endl;
			// cout<<"11111"<<endl;
			// float* dest = reinterpret_cast<float*>(state_np.get_data());
			// // cout<<"22222"<<endl;
			// for(int j=0;j<lactorState.size();j++)
			// {
			// 	dest[j] = lactorState[j];
			// }

			// cout<<"33333"<<endl;
	// time_check_start();

			p::object temp = get_la_action(state_np);
			np::ndarray action_np = np::from_object(temp);
			float* srcs = reinterpret_cast<float*>(action_np.get_data());
			for(int j=0;j<mAction.rows();j++)
			{
				mAction[j] = srcs[j];
			}
			// cout<<i<<" "<<mAction[2]<<endl;
			mActions[i] = mAction;

		}
	}
	// cout<<endl;
}


void
SLACWindow::
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
	// cout<<"00000000000)))
	// <<"

	for(int i=0;i<chars.size();i++)
	{
		// if (i!=0)
		// 	continue;
		if(chars[i]->getTeamName() == "A")
			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
		else
			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));

	}
	// cout<<"1111"<<endl;
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
	// Eigen::VectorXd mSchedulerState = mEnv->getSchedulerState(0);
	// mSubgoalCharacters[0]->getSkeleton()->setPositions(chars[0]->getSkeleton()->getPositions() + 4.0 * mSchedulerState.segment(17,2));

	// mSubgoalCharacters[0]->getSkeleton()->setPositions(chars[0]->getSkeleton()->getPositions() - mSubgoalStates[0].segment(_ID_BALL_P, 2));
	// cout<< mSubgoalStates[0].segment(_ID_BALL_P, 2).transpose()<<endl;

	// GUI::drawSkeleton(mSubgoalCharacters[0]->getSkeleton(), Eigen::Vector3d(1.0, 0.5, 0.5));


	GUI::drawSkeleton(mEnv->floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5));
	if(mEnv->mScoreBoard[0] == 1)
		GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.9, 0.3, 0.3));
	else if(mEnv->mScoreBoard[0] == 0)
		GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.3, 0.3, 0.9));
	else
		GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.1, 0.1, 0.1));

	GUI::drawSkeleton(mEnv->wallSkel, Eigen::Vector3d(0.5,0.5,0.5));
	// cout<<"2222"<<endl;

	// Not simulated just for see
	GUI::drawSkeleton(redGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));
	GUI::drawSkeleton(blueGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));
	// cout<<"3333"<<endl;

	// std::string scoreString
	// = "Red : "+to_string((int)(mEnv->mAccScore[0] + mEnv->mAccScore[1]))+" |Blue : "+to_string((int)(mEnv->mAccScore[2]+mEnv->mAccScore[3]));

	std::string scoreString
	= "Red : "+to_string((int)(mEnv->mAccScore[0]));//+" |Blue : "+to_string((int)(mEnv->mAccScore[1]));
	// cout<<"444444"<<endl;


	GUI::drawStringOnScreen(0.2, 0.8, scoreString, true, Eigen::Vector3d::Zero());

	GUI::drawStringOnScreen(0.8, 0.8, to_string(mEnv->getElapsedTime()), true, Eigen::Vector3d::Zero());


	// drawValueGradient();

	// cout<<"5555555"<<endl;




	glutSwapBuffers();
	if(mTakeScreenShot)
	{
		screenshot();
	}
	glutPostRedisplay();
}

std::string
SLACWindow::
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
SLACWindow::
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
SLACWindow::
mouse(int button, int state, int x, int y) 
{
	SimWindow::mouse(button, state, x, y);
}


void
SLACWindow::
motion(int x, int y)
{
	SimWindow::motion(x, y);
}

