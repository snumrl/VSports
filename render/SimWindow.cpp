#include "SimWindow.h"
#include "dart/external/lodepng/lodepng.h"
#include <algorithm>
#include <fstream>
#include <boost/filesystem.hpp>
#include <GL/glut.h>
using namespace GUI;
using namespace dart::simulation;
using namespace dart::dynamics;

SimWindow::
SimWindow()
:GLUTWindow(), mTakeScreenShot(false)
{
	mWorld = std::make_shared<World>();
	mDisplayTimeout = 33;
	mPlay = false;
}

void
SimWindow::
display()
{
	glClearColor(1.0, 1.0, 1, 1);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	initLights();

	mCamera->apply();

	glutSwapBuffers();
}

void
SimWindow::
keyboard(unsigned char key, int x, int y)
{
	switch(key)
	{
		case ' ':
		mPlay = !mPlay;
		break;
		case 'c':
		mTakeScreenShot ^= true;
		break;
		case 27: exit(0); break;
		default: break;
	}
}

void
SimWindow::
mouse(int button, int state, int x, int y)
{
	if((button == 3) || (button==4))
	{
		if(state == GLUT_UP) return;
		if(button == 3)
			mCamera->zoom(0,0,30,30);
		if(button == 4)
			mCamera->zoom(0,0,-30,-30);
	}

	if (state == GLUT_DOWN)
	{
		mIsDrag = true;
		mMouseType = button;
		mPrevX = x;
		mPrevY = y;
	}
	else
	{
		mIsDrag = false;
		mMouseType = 0;
	}

}

void
SimWindow::
motion(int x, int y)
{
	if (!mIsDrag)
		return;

	int mod = glutGetModifiers();
	if(mMouseType == GLUT_LEFT_BUTTON)
	{
		mCamera->rotate(x, y, mPrevX, mPrevY);
	}
	else if(mMouseType == GLUT_RIGHT_BUTTON)
	{
		mCamera->translate(x, y, mPrevX, mPrevY);
	}

	mPrevX = x;
	mPrevY = y;
	glutPostRedisplay();
}

void
SimWindow::
reshape(int w, int h)
{
	glViewport(0, 0, w, h);
	mCamera->apply();
}

void
SimWindow::
timer(int value)
{
	// if(mPlay)
	// {
	// 	// mWorld->step();
	// }
	glutTimerFunc(mDisplayTimeout, timerEvent, 1);
	// glutPostRedisplay();
}

void 
SimWindow::
screenshot() {
	static int count = 0;
	const char directory[8] = "frames";
	const char fileBase[8] = "Capture";
	char fileName[32];

	boost::filesystem::create_directories(directory);
	std::snprintf(fileName, sizeof(fileName), "%s%s%s%.4d.png",
	            directory, "/", fileBase, count++);
	int tw = glutGet(GLUT_WINDOW_WIDTH);
	int th = glutGet(GLUT_WINDOW_HEIGHT);

	glReadPixels(0, 0,  tw, th, GL_RGBA, GL_UNSIGNED_BYTE, &mScreenshotTemp[0]);

	// reverse temp2 temp1
	for (int row = 0; row < th; row++) {
		memcpy(&mScreenshotTemp2[row * tw * 4],
		&mScreenshotTemp[(th - row - 1) * tw * 4], tw * 4);
	}

	unsigned result = lodepng::encode(fileName, mScreenshotTemp2, tw, th);

	// if there's an error, display it
	if (result) {
		std::cout << "lodepng error " << result << ": "
              << lodepng_error_text(result) << std::endl;
		return ;
	} else {
		std::cout << "wrote screenshot " << fileName << "\n";
		return;
	}
}
