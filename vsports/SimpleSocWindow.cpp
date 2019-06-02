#include "SimpleSocWindow.h"
#include "../render/GLfunctionsDART.h"
#include "../model/SkelMaker.h"
#include "../model/SkelHelper.h"
#include <GL/glut.h>
#include <iostream>
using namespace dart::dynamics;
using namespace std;
double floorDepth = -0.1;

SimpleSocWindow::
SimpleSocWindow()
:SimWindow()
{
	mEnv = new Environment(30, 600, 4);
	initCustomView();
	// initCharacters();
	// initFloor();
	// initBall();
	initGoalpost();
}

void
SimpleSocWindow::
initCustomView()
{
	mCamera->eye = Eigen::Vector3d(3.60468, -4.29576, 1.87037);
	mCamera->lookAt = Eigen::Vector3d(-0.0936473, 0.158113, 0.293854);
	mCamera->up = Eigen::Vector3d(-0.132372, 0.231252, 0.963847);
}

// void 
// SimpleSocWindow::
// initFloor()
// {
// 	floorSkel = SkelHelper::makeFloor();
// 	mWorld->addSkeleton(floorSkel);
// }

// void 
// SimpleSocWindow::
// initBall()
// {
// 	ballSkel = SkelHelper::makeBall();
// 	mWorld->addSkeleton(ballSkel);
// }

void
SimpleSocWindow::
initGoalpost()
{
	redGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(-4.0, 0.0, 0.25 + floorDepth), "red");
	blueGoalpostSkel = SkelHelper::makeGoalpost(Eigen::Vector3d(4.0, 0.0, 0.25 + floorDepth), "blue");
	mWorld->addSkeleton(redGoalpostSkel);
	mWorld->addSkeleton(blueGoalpostSkel);
	// wallSkel = SkelHelper::makeWall(floorDepth);
}

// void
// SimpleSocWindow::
// initCharacters()
// {
// 	std::vector<Eigen::Vector2d> charPositions;
// 	charPositions.push_back(Eigen::Vector2d(-1.0, 0.5));
// 	charPositions.push_back(Eigen::Vector2d(-1.0, -0.5));
// 	charPositions.push_back(Eigen::Vector2d(1.0, 0.5));
// 	charPositions.push_back(Eigen::Vector2d(1.0, -0.5));
// 	// std::vector<Eigen::Vector3d> charPositions;
// 	// charPositions.push_back(Eigen::Vector3d(-1.0, floorDepth + 0.1, 0.5));
// 	// charPositions.push_back(Eigen::Vector3d(-1.0, floorDepth + 0.1, -0.5));
// 	// charPositions.push_back(Eigen::Vector3d(1.0, floorDepth + 0.1, 0.5));
// 	// charPositions.push_back(Eigen::Vector3d(1.0, floorDepth + 0.1, -0.5));


// 	for(int i=0;i<2;i++)
// 	{
// 		Character2D* character = new Character2D("red"+to_string(i));
// 		character->getSkeleton()->setPositions(charPositions[i]);
// 		charsRed.push_back(character);
// 		mWorld->addSkeleton(charsRed[i]->getSkeleton());
// 	}
// 	for(int i=0;i<2;i++)
// 	{
// 		Character2D* character = new Character2D("blue"+to_string(i));
// 		character->getSkeleton()->setPositions(charPositions[i+2]);
// 		charsBlue.push_back(character);
// 		mWorld->addSkeleton(charsBlue[i]->getSkeleton());
// 	}
// }


void
SimpleSocWindow::
keyboard(unsigned char key, int x, int y)
{
	switch(key)
	{
		case 'c':
			cout<<mCamera->eye.transpose()<<endl;
			cout<<mCamera->lookAt.transpose()<<endl;
			cout<<mCamera->up.transpose()<<endl;
		break;
		
		default: SimWindow::keyboard(key, x, y);
	}
}

void
SimpleSocWindow::
timer(int value)
{
	SimWindow::timer(value);
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


	for(int i=0;i<2;i++)
	{
		GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
	}
	for(int i=2;i<4;i++)
	{
		GUI::drawSkeleton(chars[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));
	}

	GUI::drawSkeleton(mEnv->floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5));
	GUI::drawSkeleton(mEnv->ballSkel, Eigen::Vector3d(0.1, 0.1, 0.1));
	GUI::drawSkeleton(mEnv->wallSkel, Eigen::Vector3d(0.5,0.5,0.5));




	// for(int i=0;i<charsRed.size();i++)
	// {
	// 	GUI::drawSkeleton(charsRed[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
	// }
	// for(int i=0;i<charsBlue.size();i++)
	// {
	// 	GUI::drawSkeleton(charsBlue[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));
	// }
	// GUI::drawSkeleton(floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5));
	// GUI::drawSkeleton(ballSkel, Eigen::Vector3d(0.1, 0.1, 0.1));

	// Not simulated just for see
	GUI::drawSkeleton(redGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));
	GUI::drawSkeleton(blueGoalpostSkel, Eigen::Vector3d(1.0, 1.0, 1.0));

	// // We will not draw wall
	// GUI::drawSkeleton(wallSkel, Eigen::Vector3d(0.5,0.5,0.5));


	glutSwapBuffers();
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

