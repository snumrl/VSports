#include "InteractiveWindow.h"
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
enum key_state {NOTPUSHED, PUSHED} keyarr[127];

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

void
IntWindow::
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



IntWindow::
IntWindow()
:SimWindow(), mIsNNLoaded(false)
{
	mEnv = new Environment(30, 180, 6);
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
	for(int i=0;i<mActions.size();i++)
	{
		mActions[i] = Eigen::VectorXd(4);
		mActions[i].setZero();
	}
	this->vsHardcoded = false;
// GLenum err = glewInit();
// if (err != GLEW_OK)
//   cout<<"Not ok with glew"<<endl; // or handle the error in a nicer way
// if (!GLEW_VERSION_2_1)  // check that the machine supports the 2.1 API.
//   cout<<"Not ok with glew version"<<endl; // or handle the error in a nicer way
//   cout<< glewGetString(err) <<endl; // or handle the error in a nicer way

}

IntWindow::
IntWindow(const std::string& nn_path0, const std::string& nn_path1)
:IntWindow()
{
	this->vsHardcoded = true;
	mIsNNLoaded = true;


	p::str str = ("num_state = "+std::to_string(mEnv->getNumState())).c_str();
	p::exec(str,mns);
	str = ("num_action = "+std::to_string(mEnv->getNumAction())).c_str();
	p::exec(str, mns);

	nn_module = new boost::python::object[mEnv->mNumChars];
	p::object *load = new p::object[mEnv->mNumChars];
	// reset_hidden = new boost::python::object[mEnv->mNumChars];

	for(int i=0;i<mEnv->mNumChars;i++)
	{
		nn_module[i] = p::eval("ActorCriticNN(num_state, num_action).cuda()", mns);
		load[i] = nn_module[i].attr("load");

	}
	load[0](nn_path0);
	load[1](nn_path1);
	load[2](nn_path1);
	load[3](nn_path0);
	load[4](nn_path1);
	load[5](nn_path1);

	// reset_hidden[0] = nn_module[0].attr("reset_hidden");
	// reset_hidden[1] = nn_module[1].attr("reset_hidden");
	// reset_hidden[2] = nn_module[2].attr("reset_hidden");
	// reset_hidden[3] = nn_module[3].attr("reset_hidden");

	// cout<<"3344444444"<<endl;
	// mActions.resize(mEnv->mNumChars);
	// for(int i=0;i<mActions.size();i++)
	// {
	// 	mActions[i] = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
	// }
}
IntWindow::
IntWindow(const std::string& nn_path0, const std::string& nn_path1, const std::string& nn_path2, const std::string& nn_path3)
:IntWindow()
{
	mIsNNLoaded = true;


	p::str str = ("num_state = "+std::to_string(mEnv->getNumState())).c_str();
	p::exec(str,mns);
	str = ("num_action = "+std::to_string(mEnv->getNumAction())).c_str();
	p::exec(str, mns);

	nn_module = new boost::python::object[mEnv->mNumChars];
	p::object *load = new p::object[mEnv->mNumChars];
	// reset_hidden = new boost::python::object[mEnv->mNumChars];


	for(int i=0;i<mEnv->mNumChars;i++)
	{
		nn_module[i] = p::eval("ActorCriticNN(num_state, num_action).cuda()", mns);
		load[i] = nn_module[i].attr("load");
	}

	load[0](nn_path0);
	load[1](nn_path1);
	load[2](nn_path1);
	load[3](nn_path0);
	load[4](nn_path1);
	load[5](nn_path1);


	// reset_hidden[0] = nn_module[0].attr("reset_hidden");
	// reset_hidden[1] = nn_module[1].attr("reset_hidden");
	// reset_hidden[2] = nn_module[2].attr("reset_hidden");
	// reset_hidden[3] = nn_module[3].attr("reset_hidden");
	// cout<<"3344444444"<<endl;
	// mActions.resize(mEnv->mNumChars);
	// for(int i=0;i<mActions.size();i++)
	// {
	// 	mActions[i] = Eigen::Vector4d(0.0, 0.0, 0.0, 0.0);
	// }
}
void
IntWindow::
initialize()
{
	// glewExperimental = GL_TRUE; 
	// glewInit();
	// GLuint VertexArrayID;
	// glGenVertexArrays(1, &VertexArrayID);
	// glBindVertexArray(VertexArrayID);

	// programID = loadShaders( "../vsports/shader/IntVertexShader.vertexshader", "../vsports/shader/IntVertexShader.fragmentshader" );
	// GLfloat width = 4;
	// GLfloat height = 3;
	// static const GLfloat g_vertex_buffer_data[] = {
	//   width/12.0f, height/12.0f, 0.0f,
	//    -width/12.0f, height/12.0f, 0.0f,
	//    -width/12.0f,  -height/12.0f, 0.0f,
	//    // width,  -height, 0.0f,
	// };
	// glUseProgram(programID);
	// // 이것이 우리의 버텍스 버퍼를 가리킵니다.
	// // GLuint vertexbuffer;
	// // 버퍼를 하나 생성합니다. vertexbuffer 에 결과 식별자를 넣습니다
	// glGenBuffers(1, &vertexbuffer);
	// // 아래의 명령어들은 우리의 "vertexbuffer" 버퍼에 대해서 다룰겁니다
	// glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
	// // 우리의 버텍스들을 OpenGL로 넘겨줍니다
	// glBufferData(GL_ARRAY_BUFFER, sizeof(g_vertex_buffer_data), g_vertex_buffer_data, GL_STATIC_DRAW);
}

void
IntWindow::
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
IntWindow::
initGoalpost()
{
	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-4.0, 0.0, 0.25 + floorDepth), "red");
	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(4.0, 0.0, 0.25 + floorDepth), "blue");

	mWorld->addSkeleton(redGoalpostSkel);
	mWorld->addSkeleton(blueGoalpostSkel);
}



void
IntWindow::
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
		case 'g':
			keyarr[int('g')] = PUSHED;
			break;
		case 'h':
			keyarr[int('h')] = PUSHED;
			break;
		case 'r':
			mEnv->reset();
			// for(int i=0;i<4;i++){
			// 	reset_hidden[i]();
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
IntWindow::
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
	}
}
void
IntWindow::
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
IntWindow::
applyKeyEvent()
{
	double power = 1.0;
	if(keyarr[int('w')]==PUSHED)
	{
		mActions[0][1] += power;
	}
	if(keyarr[int('s')]==PUSHED)
	{
		mActions[0][1] += -power;
	}
	if(keyarr[int('a')]==PUSHED)
	{
		mActions[0][0] += -power;
	}
	if(keyarr[int('d')]==PUSHED)
	{
		mActions[0][0] += power;
	}
	if(keyarr[int('g')]==PUSHED)
	{
		mActions[0][3] += 0.1;
	}
	else
	{
		// mActions[0][3] -= 0.1;
		// if(mActions[0][3] < -0.1)
		// 	mActions[0][3] = -0.1;
		mActions[0][3] = -0.1;
	}
	if(keyarr[int('h')]==PUSHED)
	{
		mActions[0][3] = 1.0;
		Eigen::Vector3d curVel = mEnv->getState(0);
	}
}

void
IntWindow::
step()
{
	if(mEnv->isTerminalState())
	{
		sleep(1);
		mEnv->reset();
	}
	// cout<<mEnv->getLocalState(1).segment(_ID_BALL_P,2).transpose()<<endl;

	if(mIsNNLoaded)
	{


		// if(mEnv->mNumChars == 4)
		// {
		// 	for(int i=0;i<2;i++)
		// 	{
		// 		mEnv->getState(i);
		// 		mActions[i] = mEnv->getActionFromBTree(i);

		// 	}
		// }
		if(mEnv->mNumChars == 2)
		{
			getActionFromNN(0);
			mEnv->getLocalState(1);
			mActions[1] = mEnv->getActionFromBTree(1);

		}



		if(mEnv->mNumChars == 4)
		{

			getActionFromNN(0);
			getActionFromNN(1);
			if(vsHardcoded)
			{
				for(int i=2;i<4;i++)
				{
					mEnv->getLocalState(i);
					mActions[i] = mEnv->getActionFromBTree(i);

				}
			}
			else
			{
				getActionFromNN(2);
				getActionFromNN(3);
			}
		}

		if(mEnv->mNumChars == 6)
		{

			getActionFromNN(0);
			getActionFromNN(1);
			getActionFromNN(2);
			if(vsHardcoded)
			{
				for(int i=3;i<6;i++)
				{
					mEnv->getLocalState(i);
					mActions[i] = mEnv->getActionFromBTree(i);

				}
			}
			else
			{
				getActionFromNN(3);
				getActionFromNN(4);
				getActionFromNN(5);
			}
		}


	}


	// else
	// {
	// 	if(mEnv->mNumChars == 4)
	// 	{
	// 		for(int i=0;i<4;i++)
	// 		{
	// 			mEnv->getLocalState(i);
	// 			mActions[i] = mEnv->getActionFromBTree(i);

	// 		}
	// 	// 			cout<<mEnv->getLocalState(1).transpose()<<endl;
	// 	// cout<<mEnv->getLocalState(2).transpose()<<endl;
	// 	// cout<<(mEnv->getLocalState(1)-mEnv->getLocalState(2)).transpose()<<endl;
	// 	// cout<<"##############"<<endl;
	// 	}

	// 	else if(mEnv->mNumChars == 2)
	// 	{

	// 		for(int i=0;i<2;i++)
	// 		{
	// 			mEnv->getLocalState(i);
	// 			mActions[i] = mEnv->getActionFromBTree(i);
	// 		}

	// 	}
	// }

	// cout<<"step in intWindow"<<endl;


	// applyKeyEvent();


	for(int i=0;i<mEnv->mNumChars;i++)
	{
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
IntWindow::
getActionFromNN(int index)
{
	p::object get_action;

	Eigen::VectorXd state = mEnv->getLocalState(index);

	Eigen::VectorXd mAction(mEnv->getNumAction());

	get_action = nn_module[index].attr("get_action");

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
	// cout<<"Here?"<<endl;
	mActions[index] = mAction;
	// cout<<"NO"


}

Eigen::VectorXd
IntWindow::
getValueGradient(int index)
{
	p::object get_value_gradient;
	get_value_gradient = nn_module[0].attr("get_value_gradient");

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
IntWindow::
display()
{
	glClearColor(0.85, 0.85, 1.0, 1.0);
	// glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	initLights();
	mCamera->apply();

	std::vector<Character2D*> chars = mEnv->getCharacters();


	for(int i=0;i<chars.size();i++)
	{
		// if (i!=0)
		// 	continue;
		if(chars[i]->getTeamName() == "A")
		{
			// if(i==0)
			// 	continue;
			if(mActions[i][3]>=0)
				GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.8, 0.8));
			else
				GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
		}
		else
		{
			if(mActions[i][3]>=0)
				GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.8, 0.8, 1.0));
			else
				GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));
		}

	}

	// cout<<mEnv->getLocalState(1)[_ID_SLOWED]<<endl;
	// cout<<mEnv->getLocalState(0).segment(_ID_GOALPOST_P+6, 2).transpose()<<endl;
	// cout<<endl;


	GUI::drawSkeleton(mEnv->floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5));

	GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.1, 0.1, 0.1));


	GUI::drawSkeleton(mEnv->wallSkel, Eigen::Vector3d(0.5,0.5,0.5));

	// Not simulated just for see
	GUI::drawSkeleton(redGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));
	GUI::drawSkeleton(blueGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));
	// cout<<"3333"<<endl;

	// std::string scoreString
	// = "Red : "+to_string((int)(mEnv->mAccScore[0] + mEnv->mAccScore[1]))+" |Blue : "+to_string((int)(mEnv->mAccScore[2]+mEnv->mAccScore[3]));

	std::string scoreString
	= "Red : "+to_string((int)(mEnv->mAccScore[0]));//+" |Blue : "+to_string((int)(mEnv->mAccScore[1]));
	// cout<<"444444"<<endl;

	// cout<<mEnv->getCharacters()[0]->getSkeleton()->getVelocities().transpose()<<endl;
	// cout<<mActions[1][3]<<endl;

	GUI::drawStringOnScreen(0.2, 0.8, scoreString, true, Eigen::Vector3d::Zero());

	GUI::drawStringOnScreen(0.8, 0.8, to_string(mEnv->getElapsedTime()), true, Eigen::Vector3d::Zero());
	drawValue();

	// cout<<"5555555"<<endl;


	GUI::drawSoccerLine(8, 6);

	glutSwapBuffers();
	if(mTakeScreenShot)
	{
		screenshot();
	}
	// glutPostRedisplay();
}

std::string
IntWindow::
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
IntWindow::
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

double
IntWindow::
getValue(int index)
{
	p::object get_value;
	get_value = nn_module[index].attr("get_value");

	Eigen::VectorXd state = mEnv->getState(index);
	Eigen::VectorXd value(1);

	p::tuple shape = p::make_tuple(state.size());
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape, dtype);
	float* dest = reinterpret_cast<float*>(state_np.get_data());
	for(int j=0;j<state.size();j++)
	{
		dest[j] = state[j];
	}

	p::object temp = get_value(state_np);
	np::ndarray value_np = np::from_object(temp);
	float* srcs = reinterpret_cast<float*>(value_np.get_data());
	for(int j=0;j<value.rows();j++)
	{
		value[j] = srcs[j];
	}
	return value[0];
}

void
IntWindow::
drawValue()
{
	int numChars = mEnv->mNumChars;
	// int numStates = mEnv->mStates[0].size();
	// GUI::drawStringOnScreen(0.8, 0.8, to_string(mEnv->getElapsedTime()), true, Eigen::Vector3d::Zero());

	// double leftOffset = 0.02;
	// double rightOffset = 0.04;

	Eigen::VectorXd values(numChars);

	for(int i=0;i<numChars;i++)
	{
		values[i] = getValue(i);
	}


	glPushMatrix();
	double boxSize = 1.0 / (numChars-1) / 4;

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0,0.0,1.0,
			0.0,0.0, 0.0,
			0.0, 0.0 + 1.0,0.0);

	glTranslated(-0.5, -0.5, 0.0);

	GUI::drawValueBox(values, boxSize);


	glPopMatrix();

	GLint w = glutGet(GLUT_WINDOW_WIDTH);
	GLint h = glutGet(GLUT_WINDOW_HEIGHT);

	for(int i=0;i<numChars;i++)
	{
		Eigen::Vector3d eyeToBox = Eigen::Vector3d(i * boxSize - 0.5, -0.5, 0.0);
		double fovx = mCamera->fovy * w / h;

		double boxAngleX = atan((eyeToBox[0] - boxSize/3.0)/(1.0 + boxSize)) / M_PI * 180.0;
		double boxAngleY = atan((eyeToBox[1]+boxSize)/(1.0 + boxSize)) / M_PI * 180.0; 

		// cout<<i<<endl;

		// GUI::drawStringOnScreen_small(0.5 + boxAngleX/fovx, 0.5  + boxAngleY/mCamera->fovy, indexToStateString(i), Eigen::Vector3d::Zero());

	}

}



void
IntWindow::
mouse(int button, int state, int x, int y) 
{
	SimWindow::mouse(button, state, x, y);
}


void
IntWindow::
motion(int x, int y)
{
	SimWindow::motion(x, y);
}

