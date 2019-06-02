#include "GLUTWindow.h"
#include "Camera.h"
#include <iostream>
#include "dart/external/lodepng/lodepng.h"
#include <GL/glut.h>
using namespace GUI;
std::vector<GLUTWindow*> GLUTWindow::mWindows;
std::vector<int> GLUTWindow::mWinIDs;

GLUTWindow::
GLUTWindow()
:mCamera(new Camera()), mIsDrag(false), mMouseType(0), mPrevX(0), mPrevY(0), mDisplayTimeout(1/30.0 * 1000)
{

}

GLUTWindow::
~GLUTWindow()
{

}

void
GLUTWindow::
initWindow(int _w, int _h, char* _name)
{
	mWindows.push_back(this);
	glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA | GLUT_MULTISAMPLE | GLUT_ACCUM);
	glutInitWindowPosition(1000, 100);
	glutInitWindowSize(_w, _h);
	mWinIDs.push_back(glutCreateWindow(_name));
	glutDisplayFunc(displayEvent);
	glutReshapeFunc(reshapeEvent);
	glutKeyboardFunc(keyboardEvent);
	glutMouseFunc(mouseEvent);
	glutMotionFunc(motionEvent);
	glutTimerFunc(mDisplayTimeout, timerEvent, 0);
}
inline GLUTWindow*
GLUTWindow::
current()
{
	int id = glutGetWindow();
	for(int i=0;i<mWinIDs.size();i++)
	{
		if(mWinIDs[i] == id) {
			return mWindows[i];
		}
	}
	std::cout << "GLUTWindow::current() : An unknown error occurred!" << std::endl;
	exit(0);
}

void
GLUTWindow::
displayEvent()
{
	current()->display();
}

void
GLUTWindow::
keyboardEvent(unsigned char key, int x, int y)
{
	current()->keyboard(key, x, y);
}

void
GLUTWindow::
mouseEvent(int button, int state, int x, int y)
{
	current()->mouse(button, state, x, y);
}

void
GLUTWindow::
motionEvent(int x, int y)
{
	current()->motion(x, y);
}

void
GLUTWindow::
reshapeEvent(int w, int h)
{
	current()->reshape(w, h);
}

void
GLUTWindow::
timerEvent(int value)
{
	current()->timer(value);
}


void
GLUTWindow::
initLights()
{
	static float ambient[]             = {0.3, 0.3, 0.3, 1.0};
	static float diffuse[]             = {0.2, 0.2, 0.2, 1.0};
	static float front_mat_shininess[] = {60.0};
	static float front_mat_specular[]  = {0.2, 0.2,  0.2,  1.0};
	static float front_mat_diffuse[]   = {0.2, 0.2, 0.2, 1.0};
	static float lmodel_ambient[]      = {0.2, 0.2,  0.2,  1.0};
	static float lmodel_twoside[]      = {GL_TRUE};

	GLfloat position[] = {0.0, 1.0, 1.0, 0.0};
	GLfloat position1[] = {0.0, 1.0, -1.0, 0.0};

	glEnable(GL_LIGHT0);
	glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse);
	glLightfv(GL_LIGHT0, GL_POSITION, position);

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT,  lmodel_ambient);
	glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);

	glEnable(GL_LIGHT1);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, position1);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);

	glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  front_mat_specular);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   front_mat_diffuse);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDisable(GL_CULL_FACE);
	glEnable(GL_NORMALIZE);
}