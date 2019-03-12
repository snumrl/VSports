#include "SimpleSocWindow.h"
#include "../render/GLfunctionsDART.h"
#include "../model/SkelMaker.h"
#include <GL/glut.h>
#include <iostream>
using namespace dart::dynamics;
using namespace std;
double floorDepth = -0.1;

SimpleSocWindow::
SimpleSocWindow()
:SimWindow()
{
	initCustomView();
	initCharacters();
	initFloor();
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
initFloor()
{
	floorSkel = makeFloor();
	mWorld->addSkeleton(floorSkel);
}

void
SimpleSocWindow::
initCharacters()
{
	std::vector<Eigen::Vector2d> charPositions;
	charPositions.push_back(Eigen::Vector2d(-1.0, 0.5));
	charPositions.push_back(Eigen::Vector2d(-1.0, -0.5));
	charPositions.push_back(Eigen::Vector2d(1.0, 0.5));
	charPositions.push_back(Eigen::Vector2d(1.0, -0.5));
	// std::vector<Eigen::Vector3d> charPositions;
	// charPositions.push_back(Eigen::Vector3d(-1.0, floorDepth + 0.1, 0.5));
	// charPositions.push_back(Eigen::Vector3d(-1.0, floorDepth + 0.1, -0.5));
	// charPositions.push_back(Eigen::Vector3d(1.0, floorDepth + 0.1, 0.5));
	// charPositions.push_back(Eigen::Vector3d(1.0, floorDepth + 0.1, -0.5));


	for(int i=0;i<2;i++)
	{
		Character2D* character = new Character2D("red"+to_string(i));
		character->getSkeleton()->setPositions(charPositions[i]);
		charsRed.push_back(character);
		mWorld->addSkeleton(charsRed[i]->getSkeleton());
	}
	for(int i=0;i<2;i++)
	{
		Character2D* character = new Character2D("blue"+to_string(i));
		character->getSkeleton()->setPositions(charPositions[i+2]);
		charsBlue.push_back(character);
		mWorld->addSkeleton(charsBlue[i]->getSkeleton());
	}
}


SkeletonPtr 
SimpleSocWindow::
makeFloor()
{
	SkeletonPtr floor = Skeleton::create("floor");
	Eigen::Vector3d position = Eigen::Vector3d(0.0, 0.0, floorDepth);
	Eigen::Isometry3d cb2j;
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"floor", floor, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(8.0, 6.0, 0.01),
	Eigen::Isometry3d::Identity(), cb2j);
	return floor;
}


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

	for(int i=0;i<charsRed.size();i++)
	{
		GUI::drawSkeleton(charsRed[i]->getSkeleton(), Eigen::Vector3d(1.0, 0.0, 0.0));
	}
	for(int i=0;i<charsBlue.size();i++)
	{
		GUI::drawSkeleton(charsBlue[i]->getSkeleton(), Eigen::Vector3d(0.0, 0.0, 1.0));
	}
	GUI::drawSkeleton(floorSkel, Eigen::Vector3d(0.5, 1.0, 0.5));

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

