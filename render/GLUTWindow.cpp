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
	glutKeyboardUpFunc(keyboardUpEvent);
	glutMouseFunc(mouseEvent);
	glutMotionFunc(motionEvent);
	glutTimerFunc(mDisplayTimeout, timerEvent, 0);
	mScreenshotTemp.resize(4*_w*_h);
	mScreenshotTemp2.resize(4*_w*_h);
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
keyboardUpEvent(unsigned char key, int x, int y)
{
	current()->keyboardUp(key, x, y);
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

/*
void GLUTWindow::initLights()
{
static float ambient[]             = {0.02, 0.02, 0.02, 1.0};
  static float diffuse[]             = {.1, .1, .1, 1.0};

//  static float ambient0[]            = {.01, .01, .01, 1.0};
//  static float diffuse0[]            = {.2, .2, .2, 1.0};
//  static float specular0[]           = {.1, .1, .1, 1.0};

static float ambient0[]            = {.15, .15, .15, 1.0};
static float diffuse0[]            = {.2, .2, .2, 1.0};
static float specular0[]           = {.1, .1, .1, 1.0};


  static float spot_direction0[]     = {0.0, -1.0, 0.0};


  static float ambient1[]            = {.02, .02, .02, 1.0};
  static float diffuse1[]            = {.01, .01, .01, 1.0};
  static float specular1[]           = {.01, .01, .01, 1.0};

  static float ambient2[]            = {.01, .01, .01, 1.0};
  static float diffuse2[]            = {.17, .17, .17, 1.0};
  static float specular2[]           = {.1, .1, .1, 1.0};

  static float ambient3[]            = {.06, .06, .06, 1.0};
  static float diffuse3[]            = {.15, .15, .15, 1.0};
  static float specular3[]           = {.1, .1, .1, 1.0};


  static float front_mat_shininess[] = {24.0};
  static float front_mat_specular[]  = {0.2, 0.2,  0.2,  1.0};
  static float front_mat_diffuse[]   = {0.2, 0.2, 0.2, 1.0};
  static float lmodel_ambient[]      = {0.2, 0.2,  0.2,  1.0};
  static float lmodel_twoside[]      = {GL_TRUE};

  GLfloat position0[] = {10.0, 3.0, 10.0, 1.0};
//   position0[0] = x;
//   position0[2] = z;

  GLfloat position1[] = {0.0, 1.0, -1.0, 0.0};

  GLfloat position2[] = {0.0, 5.0, 0.0, 1.0};
//   position2[0] = x-0.5;
//   position2[2] = z-0.5;

  GLfloat position3[] = {0.0, 1.3, 0.0, 1.0};
//   position3[0] = fx;
//   position3[2] = fz;

  glShadeModel(GL_SMOOTH);     
  // glClear(GL_COLOR_BUFFER_BIT);
  
  // glEnable(GL_LIGHT0);
  // glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient);
  // glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse);
  // glLightfv(GL_LIGHT0, GL_POSITION, position);

  glLightModelfv(GL_LIGHT_MODEL_AMBIENT,  lmodel_ambient);
  glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);

  glEnable(GL_LIGHT0);
  glLightfv(GL_LIGHT0, GL_AMBIENT, ambient0);
  glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuse0);
  glLightfv(GL_LIGHT0, GL_SPECULAR, specular0);
  glLightfv(GL_LIGHT0, GL_POSITION, position0);
  glLightfv(GL_LIGHT0, GL_SPOT_DIRECTION, spot_direction0);
  glLightf(GL_LIGHT0,  GL_SPOT_CUTOFF, 30.0);
  glLightf(GL_LIGHT0, GL_CONSTANT_ATTENUATION, 2.0);
  glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 1.0);
  glLightf(GL_LIGHT0, GL_QUADRATIC_ATTENUATION, 1.0);

  glEnable(GL_LIGHT1);
  glLightfv(GL_LIGHT1, GL_AMBIENT, ambient1);
  glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse1);
  glLightfv(GL_LIGHT1, GL_SPECULAR, specular1);
  glLightfv(GL_LIGHT1, GL_POSITION, position1);
  glLightf(GL_LIGHT1, GL_CONSTANT_ATTENUATION, 2.0);
  glLightf(GL_LIGHT2, GL_LINEAR_ATTENUATION, 1.0);
  glLightf(GL_LIGHT2, GL_QUADRATIC_ATTENUATION, 1.0);


  glEnable(GL_LIGHT2);
  glLightfv(GL_LIGHT2, GL_AMBIENT, ambient2);
  glLightfv(GL_LIGHT2, GL_DIFFUSE, diffuse2);
  glLightfv(GL_LIGHT2, GL_SPECULAR, specular2);
  glLightfv(GL_LIGHT2, GL_POSITION, position2);
  glLightf(GL_LIGHT2, GL_CONSTANT_ATTENUATION, 2.0);
  glLightf(GL_LIGHT2, GL_LINEAR_ATTENUATION, 1.0);
  glLightf(GL_LIGHT2, GL_QUADRATIC_ATTENUATION, 1.0);

  glEnable(GL_LIGHT3);
  glLightfv(GL_LIGHT3, GL_AMBIENT, ambient3);
  glLightfv(GL_LIGHT3, GL_DIFFUSE, diffuse3);
  glLightfv(GL_LIGHT3, GL_SPECULAR, specular3);
  glLightfv(GL_LIGHT3, GL_POSITION, position3);
  glLightf(GL_LIGHT3, GL_CONSTANT_ATTENUATION, 2.0);
  glLightf(GL_LIGHT3, GL_LINEAR_ATTENUATION, 1.0);
  glLightf(GL_LIGHT3, GL_QUADRATIC_ATTENUATION, 1.0);

  glEnable(GL_LIGHTING);
  // glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
  // glColorMaterial(GL_FRONT_AND_BACK, GL_SPECULAR);
  glEnable(GL_COLOR_MATERIAL);

  glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess);
  glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  front_mat_specular);
  glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   front_mat_diffuse);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);
  glDisable(GL_CULL_FACE);
  glEnable(GL_NORMALIZE);
}
*/
void
GLUTWindow::
initLights()
{
	static float ambient[]             = {0.4, 0.4, 0.4, 1.0};
	static float diffuse[]             = {0.2, 0.2, 0.2, 1.0};
	static float specular0[]           = {.1, .1, .1, 1.0};
	static float front_mat_shininess[] = {24.0};
	static float front_mat_specular[]  = {0.2, 0.2,  0.2,  1.0};
	static float front_mat_diffuse[]   = {0.2, 0.2, 0.2, 1.0};
	static float lmodel_ambient[]      = {0.2, 0.2,  0.2,  1.0};
	static float lmodel_twoside[]      = {GL_TRUE};



	GLfloat position[] = {0.0, 100.0, 100.0, 1.0};
	GLfloat position1[] = {0.0, 100.0, -100.0, 1.0};

  	glShadeModel(GL_SMOOTH);     
	
	glShadeModel(GL_SMOOTH);     

	glLightModelfv(GL_LIGHT_MODEL_AMBIENT,  lmodel_ambient);
	glLightModelfv(GL_LIGHT_MODEL_TWO_SIDE, lmodel_twoside);
	

	glLightfv(GL_LIGHT0, GL_AMBIENT,  ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE,  diffuse);
  	glLightfv(GL_LIGHT0, GL_SPECULAR, specular0);
	glLightfv(GL_LIGHT0, GL_POSITION, position);

	glEnable(GL_LIGHT0);

	glEnable(GL_LIGHT1);
	glLightfv(GL_LIGHT1, GL_DIFFUSE, diffuse);
	glLightfv(GL_LIGHT1, GL_POSITION, position1);
  	glLightfv(GL_LIGHT1, GL_SPECULAR, specular0);
	glEnable(GL_LIGHTING);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);

	// glMaterialfv(GL_FRONT_AND_BACK, GL_SHININESS, front_mat_shininess);
	// glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR,  front_mat_specular);
	// glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE,   front_mat_diffuse);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glDisable(GL_CULL_FACE);
	glEnable(GL_NORMALIZE);
}