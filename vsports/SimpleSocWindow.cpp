// #include "SimpleSocWindow.h"
// #include "../render/GLfunctionsDART.h"
// #include "../model/SkelMaker.h"
// #include "../model/SkelHelper.h"
// #include "../pyvs/EnvironmentPython.h"
// #include <GL/glut.h>
// #include <iostream>
// using namespace dart::dynamics;
// using namespace dart::simulation;
// using namespace std;

// namespace p = boost::python;
// namespace np = boost::python::numpy;

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

// double floorDepth = -0.1;

// SimpleSocWindow::
// SimpleSocWindow()
// :SimWindow(),vsHardcodedAI_difficulty(4.0)
// {
// 	mEnv = new Environment(30, 600, 4);
// 	initCustomView();
// 	initGoalpost();
// 	mActions = mEnv->mActions;



// 	mm = p::import("__main__");
// 	mns = mm.attr("__dict__");
// 	sys_module = p::import("sys");

// 	boost::python::str module_dir = "../pyvs";
// 	sys_module.attr("path").attr("insert")(1, module_dir);
// 	// p::exec("import os",mns);
// 	// p::exec("import sys",mns);
// 	// p::exec("import math",mns);
// 	// p::exec("import sys",mns);

// 	p::exec("import torch",mns);
// 	p::exec("import torch.nn as nn",mns);
// 	p::exec("import torch.optim as optim",mns);
// 	p::exec("import torch.nn.functional as F",mns);
// 	p::exec("import torchvision.transforms as T",mns);
// 	p::exec("import numpy as np",mns);
// 	p::exec("from Model import *",mns);
// }

// SimpleSocWindow::
// SimpleSocWindow(const std::string& nn_path)
// :SimpleSocWindow()
// {
// 	mIsNNLoaded = true;


// 	p::str str = ("num_state = "+std::to_string(mEnv->getNumState())).c_str();
// 	p::exec(str,mns);
// 	str = ("num_action = "+std::to_string(mEnv->getNumAction())).c_str();
// 	p::exec(str, mns);
// 	// str = "use_cuda = torch.cuda.is_available()";
// 	// p::exec(str, mns);


// 	// nn_module[0] = p::eval("NoCNNSimulationNN(num_state, num_action).cuda()", mns);
// 	// nn_module[1] = p::eval("NoCNNSimulationNN(num_state, num_action).cuda()", mns);

// 	// load[0] = nn_module[0].attr("load");
// 	// load[1] = nn_module[1].attr("load");

// 	// reset_hidden[0] = nn_module[0].attr("reset_hidden"); 
// 	// reset_hidden[1] = nn_module[1].attr("reset_hidden"); 
// 	// load[0](nn_path);
// 	// load[1](nn_path);
// 	nn_module = new boost::python::object[mEnv->mNumChars];
// 	p::object *load = new p::object[mEnv->mNumChars];
// 	reset_hidden = new boost::python::object[mEnv->mNumChars];

// 	for(int i=0;i<mEnv->mNumChars;i++)
// 	{
// 		nn_module[i] = p::eval("NoCNNSimulationNN(num_state, num_action).cuda()", mns);
// 		load[i] = nn_module[i].attr("load");
// 		reset_hidden[i] = nn_module[i].attr("reset_hidden");
// 		load[i](nn_path);
// 	}

// }

// void
// SimpleSocWindow::
// initCustomView()
// {
// 	mCamera->eye = Eigen::Vector3d(3.60468, -4.29576, 1.87037);
// 	mCamera->lookAt = Eigen::Vector3d(-0.0936473, 0.158113, 0.293854);
// 	mCamera->up = Eigen::Vector3d(-0.132372, 0.231252, 0.963847);
// }

// void
// SimpleSocWindow::
// initGoalpost()
// {
// 	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-4.0, 0.0, 0.25 + floorDepth), "red");
// 	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(4.0, 0.0, 0.25 + floorDepth), "blue");

// 	mWorld->addSkeleton(redGoalpostSkel);
// 	mWorld->addSkeleton(blueGoalpostSkel);
// }



// void
// SimpleSocWindow::
// keyboard(unsigned char key, int x, int y)
// {
// 	bool controlOn = false;
// 	SkeletonPtr manualSkel = mEnv->getCharacter(0)->getSkeleton();

// 	switch(key)
// 	{
// 		// case 'c':
// 		// 	cout<<mCamera->eye.transpose()<<endl;
// 		// 	cout<<mCamera->lookAt.transpose()<<endl;
// 		// 	cout<<mCamera->up.transpose()<<endl;
// 		// 	break;
// 		case 'w':
// 			if(controlOn)
// 				manualSkel->setVelocities(Eigen::Vector2d(-3.0, 0.0));
// 			break;
// 		case 's':
// 			if(controlOn)
// 				manualSkel->setVelocities(Eigen::Vector2d(3.0, 0.0));
// 			break;
// 		case 'a':
// 			if(controlOn)
// 				manualSkel->setVelocities(Eigen::Vector2d(0.0, -3.0));
// 			break;
// 		case 'd':
// 			if(controlOn)
// 				manualSkel->setVelocities(Eigen::Vector2d(0.0, 3.0));
// 			break;
// 		case 'r':
// 			mEnv->reset();
// 			reset_hidden[0]();
// 			reset_hidden[1]();
// 			reset_hidden[2]();
// 			reset_hidden[3]();
// 			break;
// 		case ']':
// 			vsHardcodedAI_difficulty += 0.1;

// 			if(vsHardcodedAI_difficulty>5.0)
// 				vsHardcodedAI_difficulty = 5.0;
// 			cout<<vsHardcodedAI_difficulty<<endl;
// 			break;
// 		case '[':
// 			vsHardcodedAI_difficulty += -0.1;
// 			if(vsHardcodedAI_difficulty<0.0)
// 				vsHardcodedAI_difficulty = 0.0;
// 			cout<<vsHardcodedAI_difficulty<<endl;
// 			break;

// 		default: SimWindow::keyboard(key, x, y);
// 	}
// }

// void
// SimpleSocWindow::
// timer(int value)
// {
// 	if(mPlay)
// 		step();
// 	SimWindow::timer(value);
// }

// void
// SimpleSocWindow::
// step()
// {
// 	if(mEnv->isTerminalState())
// 	{
// 		// for(int i=0;i<4;i++)
// 		// {
// 		// 	for(int j=mEnv->getNumState()-8;j<mEnv->getNumState();j++)
// 		// 	{
// 		// 		cout<<mEnv->getState(i)[j]<<" ";
// 		// 	}
// 		// 	cout<<endl;
// 		// }

// 		sleep(1);
// 		mEnv->reset();
// 	}
// 	// cout<<"????????"<<endl;
// 	getActionFromNN(true, true);
// 	// std::cout<<"step!"<<std::endl;
// 	for(int i=0;i<mEnv->mNumChars;i++)
// 	{
// 		// cout<<mActions[i].transpose()<<endl;
// 		// dart::collision::CollisionDetectorPtr detector = mEnv->mWorld->getConstraintSolver()->getCollisionDetector();
// 		// auto wall_char_collisionGroup = detector->createCollisionGroup(mEnv->wallSkel->getBodyNodes(), 
// 		// 	mEnv->getCharacter(i)->getSkeleton()->getRootBodyNode());
// 		// bool collision = wall_char_collisionGroup->collide();
// 		// if(collision)
// 		// {
// 		// 	mEnv->setAction(i, Eigen::VectorXd::Zero(mActions[i].size()));
// 		// 	cout<<"collide!"<<endl;
// 		// }
// 		// else
// 		mEnv->setAction(i, mActions[i]);
// 		// cout<<i<<" "<<mActions[i][2]<<endl;
// 	}

// 	mEnv->mNumIterations = 100;
// 	// int sim_per_control = mEnv->getSimulationHz()/mEnv->getControlHz();

// 	mEnv->stepAtOnce();
// 	mEnv->getRewards();
// 	// for(int i=0;i<sim_per_control;i++)
// 	// {
// 	// 	mEnv->step();
// 	// }

// }

// void
// SimpleSocWindow::
// getActionFromNN(bool vsHardcodedAI, bool isRNN)
// {
// 	// cout<<"getActionFromNN"<<endl;
// 	p::object get_action;
// 	mActions.clear();
	
// 	for(int i=0;i<mEnv->mNumChars;i++)
// 	{

// 		Eigen::VectorXd mAction(mEnv->getNumAction());
// 		std::vector<float> state = mEnv->getState(i);
// 		// mEnv->getState(i);
// 		// Eigen::VectorXd state = mEnv->mStates[i];
// 		//change the i=1 agent
// 		if(vsHardcodedAI && (i == 2 || i == 3))
// 		{
// 			// cout<<"i : "<<i<<endl;
// 			Eigen::VectorXd curBallRelaltionalP = mEnv->mStates[i].segment(ID_BALL_P,2);
// 			Eigen::VectorXd direction = curBallRelaltionalP.normalized();
// 			Eigen::VectorXd curVel = mEnv->mStates[i].segment(ID_V,2);
// 			mAction.segment(0, 2) = (direction*vsHardcodedAI_difficulty - curVel);

// 			for(int j=0;j<2;j++)
// 			{
// 				if(mAction[j] > 0.5)
// 					mAction[j] = 0.5;
// 				else if(mAction[j] < -0.5)
// 					mAction[j] = -0.5;
// 			}

// 			// mAction[2] = rand()%3-1;
// 			mAction[2] = 1.0;
// 			// mAction[2] = 0;
// 			mActions.push_back(mAction);
// 		}
// 		else
// 		{
// 			if(!isRNN)
// 				get_action = nn_module[i].attr("get_action");
// 			else
// 				get_action = nn_module[i].attr("get_action_rnn");
// 			// cout<<"i : "<<i<<endl;
// 			p::tuple shape = p::make_tuple(state.size());
// 			np::dtype dtype = np::dtype::get_builtin<float>();
// 			np::ndarray state_np = np::empty(shape, dtype);

// 			// cout<<state.segment(0,6).transpose()<<endl;
// 			// cout<<shape<<endl;
// 			// cout<<"11111"<<endl;
// 			float* dest = reinterpret_cast<float*>(state_np.get_data());
// 			// cout<<"22222"<<endl;
// 			for(int j=0;j<state.size();j++)
// 			{
// 				dest[j] = state[j];
// 			}

// 			// cout<<"33333"<<endl;
// 	// time_check_start();

// 			p::object temp = get_action(state_np);
// 			np::ndarray action_np = np::from_object(temp);
// 			float* srcs = reinterpret_cast<float*>(action_np.get_data());
// 			for(int j=0;j<mAction.rows();j++)
// 			{
// 				mAction[j] = srcs[j];
// 			}
// 			// cout<<i<<" "<<mAction[2]<<endl;
// 			mActions.push_back(mAction);

// 		}
// 	}
// 	// cout<<endl;
// }


// void
// SimpleSocWindow::
// display()
// {
// 	glClearColor(0.85, 0.85, 1.0, 1.0);
// 	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
// 	glEnable(GL_DEPTH_TEST);
// 	initLights();
// 	mCamera->apply();

// 	std::vector<Character2D*> chars = mEnv->getCharacters();

// 	// exit(0);
// 	// mEnv->getState_map(0);
// 	// exit(0);

// 	for(int i=0;i<chars.size();i++)
// 	{
// 		if(chars[i]->getTeamName() == "A")
// 			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
// 		else
// 			GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));


// 	}
// 	// GUI::drawSkeleton(chars[0]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
// 	// GUI::drawSkeleton(chars[1]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));


// 	// for(int i=0;i<2;i++)
// 	// {
// 	// 	GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
// 	// }
// 	// for(int i=2;i<4;i++)
// 	// {
// 	// 	GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));
// 	// }

// 	GUI::drawSkeleton(mEnv->floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5));
// 	if(mEnv->mScoreBoard[0] == 1)
// 		GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.9, 0.3, 0.3));
// 	else if(mEnv->mScoreBoard[0] == 0)
// 		GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.3, 0.3, 0.9));
// 	else
// 		GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.1, 0.1, 0.1));

// 	GUI::drawSkeleton(mEnv->wallSkel, Eigen::Vector3d(0.5,0.5,0.5));

// 	// Not simulated just for see
// 	GUI::drawSkeleton(redGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));
// 	GUI::drawSkeleton(blueGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));

// 	std::string scoreString
// 	= "Red : "+to_string((int)(mEnv->mAccScore[0] + mEnv->mAccScore[1]))+" |Blue : "+to_string((int)(mEnv->mAccScore[2]+mEnv->mAccScore[3]));



// 	GUI::drawStringOnScreen(0.2, 0.8, scoreString, true, Eigen::Vector3d::Zero());

// 	GUI::drawStringOnScreen(0.8, 0.8, to_string(mEnv->getElapsedTime()), true, Eigen::Vector3d::Zero());


// 	// GUI::drawMapOnScreen(mEnv->mMapStates[0]->minimaps[0], 84, 84);


// 	glutSwapBuffers();
// 	if(mTakeScreenShot)
// 	{
// 		screenshot();
// 	}
// 	glutPostRedisplay();
// }

// void
// SimpleSocWindow::
// mouse(int button, int state, int x, int y) 
// {
// 	SimWindow::mouse(button, state, x, y);
// }


// void
// SimpleSocWindow::
// motion(int x, int y)
// {
// 	SimWindow::motion(x, y);
// }

