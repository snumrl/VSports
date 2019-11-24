#include "AgentEnvWindow.h"
#include "Environment.h"
#include "../render/GLfunctionsDART.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include "../pyvs/EnvironmentPython.h"
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

AgentEnvWindow::
AgentEnvWindow(int index, Environment* env)
:SimWindow(), mIndex(index)
{
	mEnv = env;
	initWindow(96, 72, "Agent 0 view");
	setAgentView();
	initGoalpost();

}

void
AgentEnvWindow::
initWindow(int _w, int _h, char* _name)
{
	mWindows.push_back(this);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE | GLUT_ACCUM);
	glutInitWindowPosition(1500, -500);
	glutInitWindowSize(_w, _h);
	mWinIDs.push_back(glutCreateWindow(_name));
	// glutHideWindow();
	// glutDisplayFunc(displayEvent);
	// glutReshapeFunc(reshapeEvent);
	// glutKeyboardFunc(keyboardEvent);
	// glutKeyboardUpFunc(keyboardUpEvent);
	// glutMouseFunc(mouseEvent);
	// glutMotionFunc(motionEvent);
	// glutTimerFunc(mDisplayTimeout, timerEvent, 0);
	mScreenshotTemp.resize(4*_w*_h);
	mScreenshotTemp2.resize(4*_w*_h);

	agentViewImg.resize(4*_w*_h);
	agentViewImgTemp.resize(4*_w*_h);

	// glfw.window_hint(glfw.VISIBLE, false);
	// glut.
}



void
AgentEnvWindow::
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

}

void
AgentEnvWindow::
initGoalpost()
{
	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-4.0, 0.0, 0.25 + floorDepth), "red");
	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(4.0, 0.0, 0.25 + floorDepth), "blue");

	mWorld->addSkeleton(redGoalpostSkel);
	mWorld->addSkeleton(blueGoalpostSkel);
}

void
AgentEnvWindow::
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



	GUI::drawSoccerLine(8, 6);

	// glutSwapBuffers();
	// if(mTakeScreenShot)
	// {
	// 	screenshot();
	// }
	glutPostRedisplay();

	getAgentView();
	// screenshot();
}

void
AgentEnvWindow::
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


void 
AgentEnvWindow::
keyboard(unsigned char key, int x, int y)
{
	// SimWindow::keyboard(key, x, y);
}
void 
AgentEnvWindow::
keyboardUp(unsigned char key, int x, int y)
{
	// SimWindow::keyboardUp(key, x, y);
}
void 
AgentEnvWindow::
timer(int value)
{
	SimWindow::timer(value);
}
void 
AgentEnvWindow::
mouse(int button, int state, int x, int y)
{

}
void 
AgentEnvWindow::
motion(int x, int y)
{

}
