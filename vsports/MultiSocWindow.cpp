#include "MultiSocWindow.h"
#include "../render/GLfunctionsDART.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include <GL/glut.h>
#include <iostream>
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

MultiSocWindow::
MultiSocWindow()
:SimWindow()
{
	mEnv = new Environment(30, 600, 2);
	initCustomView();
	initGoalpost();
	mActions = mEnv->mActions;


	mm = p::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = p::import("sys");

	boost::python::str module_dir = "../pyvs";
	sys_module.attr("path").attr("insert")(1, module_dir);

	p::exec("import torch",mns);
	p::exec("import torch.nn as nn",mns);
	p::exec("import torch.optim as optim",mns);
	p::exec("import torch.nn.functional as F",mns);
	p::exec("import torchvision.transforms as T",mns);
	p::exec("import numpy as np",mns);
	p::exec("from Model import *", mns);
}

MultiSocWindow::
MultiSocWindow(char** paths)
:MultiSocWindow()
{
	mIsNNLoaded = true;

	p::str str = ("num_state = "+std::to_string(mEnv->getNumState())).c_str();
	p::exec(str,mns);
	str = ("num_action = "+std::to_string(mEnv->getNumAction())).c_str();
	p::exec(str, mns);


	vector<p::object> loads;
	// cout<<"11111"<<endl;
	for(int i=0;i<2;i++)
	{
	// cout<<"2222"<<endl;
		nn_modules.push_back(p::eval("SimulationNN(num_state, num_action)", mns));

	// cout<<"3333"<<endl;
		loads.push_back(nn_modules[i].attr("load"));
	// cout<<"4444"<<endl;
		loads[i](string(paths[i+1]));
	}

}

void
MultiSocWindow::
initCustomView()
{
	mCamera->eye = Eigen::Vector3d(3.60468, -4.29576, 1.87037);
	mCamera->lookAt = Eigen::Vector3d(-0.0936473, 0.158113, 0.293854);
	mCamera->up = Eigen::Vector3d(-0.132372, 0.231252, 0.963847);
}

void
MultiSocWindow::
initGoalpost()
{
	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-4.0, 0.0, 0.25 + floorDepth), "red");
	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(4.0, 0.0, 0.25 + floorDepth), "blue");

	mWorld->addSkeleton(redGoalpostSkel);
	mWorld->addSkeleton(blueGoalpostSkel);
}



void
MultiSocWindow::
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
		
		default: SimWindow::keyboard(key, x, y);
	}
}

void
MultiSocWindow::
timer(int value)
{
	if(mPlay)
		step();
	SimWindow::timer(value);
}

void
MultiSocWindow::
step()
{
	getActionFromNN();
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

	int sim_per_control = mEnv->getSimulationHz()/mEnv->getControlHz();
	for(int i=0;i<sim_per_control;i++)
	{
		mEnv->step();
	}

}

void
MultiSocWindow::
getActionFromNN()
{
	p::object get_action;
	mActions.clear();
	for(int i=0;i<mEnv->mNumChars;i++)
	{
		get_action = nn_modules[i].attr("get_action");
		Eigen::VectorXd mAction(mEnv->getNumAction());
		std::vector<double> state = mEnv->getState(i);
		// mEnv->getState(i);
		// Eigen::VectorXd state = mEnv->mStates[i];
		p::tuple shape = p::make_tuple(state.size());
		np::dtype dtype = np::dtype::get_builtin<float>();
		np::ndarray state_np = np::empty(shape, dtype);

		float* dest = reinterpret_cast<float*>(state_np.get_data());
		for(int j=0;j<state.size();j++)
		{
			dest[j] = state[j];
		}

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


void
MultiSocWindow::
display()
{
	glClearColor(0.85, 0.85, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	initLights();
	mCamera->apply();

	std::vector<Character2D*> chars = mEnv->getCharacters();



	GUI::drawSkeleton(chars[0]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
	GUI::drawSkeleton(chars[1]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));


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
	= "Red : "+to_string((int)mEnv->mAccScore[0])+" |Blue : "+to_string((int)mEnv->mAccScore[1]);



	GUI::drawStringOnScreen(0.2, 0.8, scoreString, true, Eigen::Vector3d::Zero());


	glutSwapBuffers();
	if(mTakeScreenShot)
	{
		screenshot();
	}
	glutPostRedisplay();
}

void
MultiSocWindow::
mouse(int button, int state, int x, int y) 
{
	SimWindow::mouse(button, state, x, y);
}


void
MultiSocWindow::
motion(int x, int y)
{
	SimWindow::motion(x, y);
}
