#include "EpisodeReplayWindow.h"
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
#include <unistd.h>

using namespace dart::dynamics;
using namespace dart::simulation;
using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;
// enum key_state {NOTPUSHED, PUSHED} keyarr[127];

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

void
EpiWindow::
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



EpiWindow::
EpiWindow()
:SimWindow()
{
	mEnv = new Environment(30, 180, 6);
	initCustomView();
	initGoalpost();
}

EpiWindow::
EpiWindow(const std::string& replay_path, const std::string& numPath)
:EpiWindow()
{
	mNumPath = atoi(numPath.c_str());
	for(int i=0;i<mNumPath;i++)
	{
		mReplayPathList.push_back(replay_path +"_"+ to_string(i)+".txt");
	}
	for(int i=0;i<mNumPath;i++)
		cout<<mReplayPathList[i]<<endl;


	curPathIndex = 0;
	curTimeStep = 0;
	curTD = 0.0; // To show TD precisely
	storeEpisodeFromPath();

}

void
EpiWindow::
storeEpisodeFromPath()
{
	int numStates = mEnv->getNumState();
	for(int i=0;i<mNumPath;i++)
	{
		std::vector<Eigen::VectorXd> states;
		std::vector<double> TDs;
		int epiLength;
		ifstream in;
		in.open(mReplayPathList[i]);
		in >> epiLength;

		for(int t=0;t<epiLength;t++)
		{
			string bracket;
			// in >> bracket;
			Eigen::VectorXd state(numStates);
			double TD;
			for(int j=0;j<numStates;j++)
			{
				in >> state[j];
			}
			// in >> bracket;
			in >> TD;

			states.push_back(state);
			TDs.push_back(TD);
		}
		in.close();

		stateList.push_back(states);
		TDList.push_back(TDs);
		// cout<<state.transpose()<<endl;
	}

	// for(int t=0;t<stateList[9].size();t++)
	// {
	// 	cout<<stateList[9][t].transpose()<<endl;
	// }
}

void
EpiWindow::
reconEnvFromCurrentState()
{
	mEnv->reconEnvFromState(0, stateList[curPathIndex][curTimeStep]);
	// Eigen::VectorXd curState = localStateToOriginState(stateList[curPathIndex][curTimeStep],2);
	// double curTD = TDList[curPathIndex][curTimeStep];

	// mEnv->getCharacter(0)->getSkeleton()->setPosition(0, 8.0*curState[_ID_P]);
	// mEnv->getCharacter(0)->getSkeleton()->setPosition(1, 8.0*curState[_ID_P+1]);
	// Eigen::Vector2d p = curState.segment(_ID_P, 2);
	// Eigen::Vector2d temp;
	// if(mEnv->mNumChars == 6)
	// {

	// 	// cout<<curState.transpose()<<endl;
	// 	temp = curState.segment(_ID_ALLY1_P, 2);
	// 	// temp = rotate2DVector(temp, theta);

	// 	mEnv->getCharacter(1)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
	// 	mEnv->getCharacter(1)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);
		

	// 	temp = curState.segment(_ID_ALLY2_P, 2);
	// 	// temp = rotate2DVector(temp, theta);

	// 	mEnv->getCharacter(2)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
	// 	mEnv->getCharacter(2)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);


	// 	temp = curState.segment(_ID_OP_DEF_P, 2);
	// 	// temp = rotate2DVector(temp, theta);

	// 	mEnv->getCharacter(3)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
	// 	mEnv->getCharacter(3)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

	// 	temp = curState.segment(_ID_OP_ATK1_P, 2);
	// 	// temp = rotate2DVector(temp, theta);

	// 	mEnv->getCharacter(4)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
	// 	mEnv->getCharacter(4)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);

	// 	temp = curState.segment(_ID_OP_ATK2_P, 2);
	// 	// temp = rotate2DVector(temp, theta);

	// 	mEnv->getCharacter(5)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
	// 	mEnv->getCharacter(5)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);
	// }

	// else if(mEnv->mNumChars == 2)
	// {
	// 	// cout<<curState.transpose()<<endl;
	// 	temp = curState.segment(_ID_OP_P, 2);
	// 	// temp = rotate2DVector(temp, theta);

	// 	mEnv->getCharacter(1)->getSkeleton()->setPosition(0, 8.0*(p+temp)[0]);
	// 	mEnv->getCharacter(1)->getSkeleton()->setPosition(1, 8.0*(p+temp)[1]);
		

	// }

	



	// temp = curState.segment(_ID_BALL_P, 2);
	// // temp = rotate2DVector(temp, theta);

	// mEnv->ballSkel->setPosition(0, 8.0*(p+temp)[0]);
	// mEnv->ballSkel->setPosition(1, 8.0*(p+temp)[1]);



	// mEnv->getCharacter(3)->getSkeleton()->setPosition(0, 8.0*curState[_ID_OP_ATK_P]);
	// mEnv->getCharacter(3)->getSkeleton()->setPosition(1, 8.0*curState[_ID_OP_ATK_P+1]);

}

void
EpiWindow::
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
EpiWindow::
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
EpiWindow::
initGoalpost()
{
	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-4.0, 0.0, 0.25 + floorDepth), "red");
	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(4.0, 0.0, 0.25 + floorDepth), "blue");

	mWorld->addSkeleton(redGoalpostSkel);
	mWorld->addSkeleton(blueGoalpostSkel);
}



void
EpiWindow::
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

		case 'r':
			mEnv->reset();
			// reset_hidden[2]();
			// reset_hidden[3]();
			break;
		case 'l':
			controlOn = !controlOn;
			break;

		case 'p':
			curTimeStep += 10;
		case ']':
			curTimeStep++;
			if(curTimeStep >= stateList[curPathIndex].size())
			{
				// sleep(1);
				curTimeStep = 0;
				curPathIndex++;
				if(curPathIndex>=mNumPath)
				{
					curPathIndex = 0;
				}
			}
			curTD = TDList[curPathIndex][curTimeStep];
			reconEnvFromCurrentState();
			break;
		case 'o':
			curTimeStep -= 10;
		case '[':
			curTimeStep--;
			if(curTimeStep < 0)
			{
				if(curPathIndex == 0)
				{
					curPathIndex = mNumPath-1;
					curTimeStep = stateList[curPathIndex].size()-1;
				}
				else
				{
					curPathIndex--;
					curTimeStep = stateList[curPathIndex].size()-1;
				}
			}
			curTD = TDList[curPathIndex][curTimeStep];
			reconEnvFromCurrentState();
			break;

		default: SimWindow::keyboard(key, x, y);
	}
}
void
EpiWindow::
keyboardUp(unsigned char key, int x, int y)
{
	SkeletonPtr manualSkel = mEnv->getCharacter(0)->getSkeleton();

	switch(key)
	{
		// case 'w':
		// 	keyarr[int('w')] = NOTPUSHED;
		// 	break;
		// case 's':
		// 	keyarr[int('s')] = NOTPUSHED;
		// 	break;
		// case 'a':
		// 	keyarr[int('a')] = NOTPUSHED;
		// 	break;
		// case 'd':
		// 	keyarr[int('d')] = NOTPUSHED;
		// 	break;
		// case 'g':
		// 	keyarr[int('g')] = NOTPUSHED;
		// 	break;
		// case 'h':
		// 	keyarr[int('h')] = NOTPUSHED;
		// 	break;
	}
}
void
EpiWindow::
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
EpiWindow::
step()
{
	reconEnvFromCurrentState();
	// if(abs(TDList[curPathIndex][curTimeStep])>0.1)
	// 	usleep(50000);

	curTD = TDList[curPathIndex][curTimeStep];

	curTimeStep++;
	if(curTimeStep >= stateList[curPathIndex].size())
	{
		// sleep(1);
		curTimeStep = 0;
		curPathIndex++;
		if(curPathIndex>=mNumPath)
		{
			curPathIndex = 0;
		}

	}


}

void
EpiWindow::
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
			// if(mActions[i][4]>=0)
			// 	GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.8, 0.8));
			// else
				GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
		}
		else
		{
			// if(mActions[i][4]>=0)
			// 	GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.8, 0.8, 1.0));
			// else
				GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));
		}

	}

	// cout<<mEnv->getLocalState(0).segment(_ID_GOALPOST_P+4, 2).transpose()<<endl;
	// cout<<mEnv->getLocalState(0).segment(_ID_GOALPOST_P+6, 2).transpose()<<endl;
	// cout<<endl;


	GUI::drawSkeleton(mEnv->floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5));

	GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.1, 0.1, 0.1));


	GUI::drawSkeleton(mEnv->wallSkel, Eigen::Vector3d(0.5,0.5,0.5));

	// Not simulated just for see
	GUI::drawSkeleton(redGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));
	GUI::drawSkeleton(blueGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));
	// cout<<"3333"<<endl;


	std::string scoreString
	= "Red : "+to_string(curTD);//+" |Blue : "+to_string((int)(mEnv->mAccScore[1]));

	GUI::drawStringOnScreen(0.2, 0.8, scoreString, true, Eigen::Vector3d::Zero());

	GUI::drawStringOnScreen(0.8, 0.8, to_string(mEnv->getElapsedTime()), true, Eigen::Vector3d::Zero());
	// drawValue();

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
EpiWindow::
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


// double
// EpiWindow::
// getValue(int index)
// {
// 	p::object get_value;
// 	get_value = nn_module[index].attr("get_value");

// 	Eigen::VectorXd state = mEnv->getState(index);
// 	Eigen::VectorXd value(1);

// 	p::tuple shape = p::make_tuple(state.size());
// 	np::dtype dtype = np::dtype::get_builtin<float>();
// 	np::ndarray state_np = np::empty(shape, dtype);
// 	float* dest = reinterpret_cast<float*>(state_np.get_data());
// 	for(int j=0;j<state.size();j++)
// 	{
// 		dest[j] = state[j];
// 	}

// 	p::object temp = get_value(state_np);
// 	np::ndarray value_np = np::from_object(temp);
// 	float* srcs = reinterpret_cast<float*>(value_np.get_data());
// 	for(int j=0;j<value.rows();j++)
// 	{
// 		value[j] = srcs[j];
// 	}
// 	return value[0];
// }

// void
// EpiWindow::
// drawValue()
// {
// 	int numChars = mEnv->mNumChars;
// 	// int numStates = mEnv->mStates[0].size();
// 	// GUI::drawStringOnScreen(0.8, 0.8, to_string(mEnv->getElapsedTime()), true, Eigen::Vector3d::Zero());

// 	// double leftOffset = 0.02;
// 	// double rightOffset = 0.04;

// 	Eigen::VectorXd values(numChars);

// 	for(int i=0;i<numChars;i++)
// 	{
// 		values[i] = getValue(i);
// 	}


// 	glPushMatrix();
// 	double boxSize = 1.0 / (numChars-1) / 4;

// 	glMatrixMode(GL_MODELVIEW);
// 	glLoadIdentity();
// 	gluLookAt(0.0,0.0,1.0,
// 			0.0,0.0, 0.0,
// 			0.0, 0.0 + 1.0,0.0);

// 	glTranslated(-0.5, -0.5, 0.0);

// 	GUI::drawValueBox(values, boxSize);


// 	glPopMatrix();

// 	GLint w = glutGet(GLUT_WINDOW_WIDTH);
// 	GLint h = glutGet(GLUT_WINDOW_HEIGHT);

// 	for(int i=0;i<numChars;i++)
// 	{
// 		Eigen::Vector3d eyeToBox = Eigen::Vector3d(i * boxSize - 0.5, -0.5, 0.0);
// 		double fovx = mCamera->fovy * w / h;

// 		double boxAngleX = atan((eyeToBox[0] - boxSize/3.0)/(1.0 + boxSize)) / M_PI * 180.0;
// 		double boxAngleY = atan((eyeToBox[1]+boxSize)/(1.0 + boxSize)) / M_PI * 180.0; 

// 		// cout<<i<<endl;

// 		// GUI::drawStringOnScreen_small(0.5 + boxAngleX/fovx, 0.5  + boxAngleY/mCamera->fovy, indexToStateString(i), Eigen::Vector3d::Zero());

// 	}

// }



void
EpiWindow::
mouse(int button, int state, int x, int y) 
{
	SimWindow::mouse(button, state, x, y);
}


void
EpiWindow::
motion(int x, int y)
{
	SimWindow::motion(x, y);
}

