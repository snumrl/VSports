#include "GLfunctions.h"
#include <assimp/cimport.h>
#include <iostream>
// #include <GL/gl.h>
// #include <GL/glu.h>
#include <GL/glut.h>
#include <math.h>

using namespace std;

static GLUquadricObj *quadObj;
static void initQuadObj(void)
{
	quadObj = gluNewQuadric();
	if(!quadObj)
    	// DART modified error output
    	std::cerr << "OpenGL: Fatal Error in DART: out of memory." << std::endl;
}
#define QUAD_OBJ_INIT { if(!quadObj) initQuadObj();}

void
GUI::
drawSphere(double r)
{
	QUAD_OBJ_INIT;
	gluQuadricDrawStyle(quadObj, GLU_FILL);
	gluQuadricNormals(quadObj, GLU_SMOOTH);

	gluSphere(quadObj, r, 16, 16);
}

void
GUI::
drawCube(const Eigen::Vector3d& _size)
{
	glScaled(_size[0], _size[1], _size[2]);

	// Code taken from glut/lib/glut_shapes.c
    static GLfloat n[6][3] =
    {
        {-1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0},
        {0.0, -1.0, 0.0},
        {0.0, 0.0, 1.0},
        {0.0, 0.0, -1.0}
    };
    static GLint faces[6][4] =
    {
        {0, 1, 2, 3},
        {3, 2, 6, 7},
        {7, 6, 5, 4},
        {4, 5, 1, 0},
        {5, 6, 2, 1},
        {7, 4, 0, 3}
    };
    GLfloat v[8][3];
    GLint i;
    GLfloat size = 1;

    v[0][0] = v[1][0] = v[2][0] = v[3][0] = -size / 2;
    v[4][0] = v[5][0] = v[6][0] = v[7][0] = size / 2;
    v[0][1] = v[1][1] = v[4][1] = v[5][1] = -size / 2;
    v[2][1] = v[3][1] = v[6][1] = v[7][1] = size / 2;
    v[0][2] = v[3][2] = v[4][2] = v[7][2] = -size / 2;
    v[1][2] = v[2][2] = v[5][2] = v[6][2] = size / 2;

    for (i = 5; i >= 0; i--) {
        glBegin(GL_QUADS);
        glNormal3fv(&n[i][0]);
        glVertex3fv(&v[faces[i][0]][0]);
        glVertex3fv(&v[faces[i][1]][0]);
        glVertex3fv(&v[faces[i][2]][0]);
        glVertex3fv(&v[faces[i][3]][0]);
        glEnd();
    }
}

void
GUI::
drawTriangle(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& p2,const Eigen::Vector3d& color)
{
	glColor3f(color[0], color[1], color[2]);
	glBegin(GL_TRIANGLES);
	glVertex3d(p0[0], p0[1], p0[2]);
	glVertex3d(p1[0], p1[1], p1[2]);
	glVertex3d(p2[0], p2[1], p2[2]);
	glEnd();
}

void
GUI::
drawSquare(double width, double height)
{
	glColor3f(1.0, 1.0, 1.0);
	glBegin(GL_POLYGON);
	glNormal3f(0.0, 0.0, 1.0);
	glVertex3f(width, height, 0);
	glVertex3f(-width, height, 0);
	glVertex3f(-width, -height, 0);
	glVertex3f(width, -height, 0);
	glEnd();
}

void
GUI::
drawTetrahedron(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& p2,const Eigen::Vector3d& p3,const Eigen::Vector3d& color)
{
	drawTriangle(p0,p1,p2,color);
	drawTriangle(p0,p1,p3,color);
	drawTriangle(p0,p2,p3,color);
	drawTriangle(p1,p2,p3,color);
}

void
GUI::
drawLine(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& color)
{
	glColor3f(color[0],color[1],color[2]);
	glBegin(GL_LINES);
	glVertex3f(p0[0],p0[1],p0[2]);
	glVertex3f(p1[0],p1[1],p1[2]);
	glEnd();
}

void
GUI::
drawPoint(const Eigen::Vector3d& p0,const Eigen::Vector3d& color)
{
	glColor3f(color[0],color[1],color[2]);
	glBegin(GL_POINTS);
	glVertex3f(p0[0],p0[1],p0[2]);
	glEnd();
}

void
GUI::drawStringOnScreen(float _x, float _y, const std::string &_s, bool _bigFont, const Eigen::Vector3d& color)
{
	glColor3f(color[0], color[1], color[2]);

	// draws text on the screen
	GLint oldMode;
	glGetIntegerv(GL_MATRIX_MODE, &oldMode);
	glMatrixMode(GL_PROJECTION);

	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glRasterPos2f(_x, _y);

	unsigned int length = _s.length();
	for(unsigned int c =0; c<length; c++)
	{
		if(_bigFont)
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, _s.at(c));
		else
			glutBitmapCharacter(GLUT_BITMAP_HELVETICA_10, _s.at(c));
	}
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(oldMode);
}

void
GUI::drawStringOnScreen_small(float _x, float _y, const std::string &_s, const Eigen::Vector3d& color)
{
	glColor3f(color[0], color[1], color[2]);

	// draws text on the screen
	GLint oldMode;
	glGetIntegerv(GL_MATRIX_MODE, &oldMode);
	glMatrixMode(GL_PROJECTION);

	glPushMatrix();
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	glRasterPos2f(_x, _y);

	unsigned int length = _s.length();
	for(unsigned int c =0; c<length; c++)	
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, _s.at(c));
	}
	glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();
	glMatrixMode(oldMode);
}


void
GUI::drawMapOnScreen(Eigen::VectorXd minimap, int numRows, int numCols)
{
	// draws text on the screen
	double _x, _y;
	_x = 0.8;
	_y = 0.0;
	GLint oldMode;
	glGetIntegerv(GL_MATRIX_MODE, &oldMode);
	glMatrixMode(GL_PROJECTION);

	glPushMatrix();
	glDisable(GL_LIGHTING);
	glLoadIdentity();
	gluOrtho2D(0.0, 1.0, 0.0, 1.0);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadIdentity();
	// glRasterPos2f(_x, _y);

	for(int col=0;col<numCols;col++)
	{
		for(int row=0;row<numRows;row++)
		{
			glPushMatrix();
			glLoadIdentity();
			double curColor = minimap[col*numRows+row];
			glTranslated(_x+0.2*row/(double)numRows, _y+(0.2-0.2*col/(double)numCols), 0.0);

			glColor3f(curColor, curColor, curColor);
			drawCube(Eigen::Vector3d(0.0025, 0.0025, 0.0025));
			glTranslated(-_x-0.2*row/(double)numRows, -_y-(0.2-0.2*col/(double)numCols), 0.0);
			glPopMatrix();

			// glTranslated(-0.5-0.4*row/numRows, -0.5-0.4*col/numCols, 0.0);
			// glTranslated(-0.5-0.01*row/numCols, -0.5-0.01*col/numRows, 0.0);

		}
	}
	// glTranslated(0.1, 0.1, 0.0);
	// drawCube(Eigen::Vector3d(0.05, 0.05, 0.05));



	glPopMatrix();

	// glPushMatrix();
	// glLoadIdentity();
	// glTranslated(0.2, 0.2, 0.0);
	// drawCube(Eigen::Vector3d(0.05, 0.05, 0.05));
	// glPopMatrix();

	glMatrixMode(GL_PROJECTION);
	glEnable(GL_LIGHTING);
	glPopMatrix();
	glMatrixMode(oldMode);
}

Eigen::Vector3d degreeToRgb(double degree, bool forValue)
{
	Eigen::Vector3d r(1.0, 0.0, 0.0);
	Eigen::Vector3d g(0.0, 1.0, 0.0);
	Eigen::Vector3d b(0.0, 0.0, 1.0);
	Eigen::Vector3d resultRgb;

	if(forValue)
	{
		if(degree>1.0)
			degree = 1.0;
		if(degree<-1.0)
			degree= -1.0;

		degree = degree/2.0+0.5;
		resultRgb = Eigen::Vector3d::Ones()*degree;
	}
	else
	{
		degree = abs(degree);
		// std::cout<<degree<<" "<<std::abs(degree)<<std::endl;

		if(degree>=1.0)
			degree = 1.0;


		int option = 1;

		if(option == 0)
		{
			if(degree<=1.0/2.0)
			{
				double iFactor = degree * 2.0;
				resultRgb = degree * g + (1.0 - degree) * b;
			}
			else
			{
				double iFactor = (degree-1.0/2.0) * 2.0;
				resultRgb = degree * r + (1.0 - degree) * g;
			}
		}
		
		else if(option == 1)
		{
			resultRgb = Eigen::Vector3d::Ones()*degree;
		}
		

	}


	return resultRgb;
}

void 
GUI::drawValueGradientBox(Eigen::VectorXd states, Eigen::VectorXd valueGradient, double boxSize)
{
	assert(states.size() == valueGradient.size());
	int stateSize = states.size();

	Eigen::Vector3d boxSizeVector(boxSize, boxSize, boxSize);
		// std::cout<<valueGradient.transpose()<<std::endl;

	for(int i=0;i<stateSize;i++)
	{
		glPushMatrix();
		Eigen::Vector3d color = degreeToRgb(valueGradient[i]/10.0);
		glColor3f(color[0], color[1], color[2]);
		glTranslated(boxSize*i, 0, 0);
		drawCube(boxSizeVector);
		glPopMatrix();
	}
}

void 
GUI::drawValueBox(Eigen::VectorXd value, double boxSize)
{
	int valueSize = value.size();

	Eigen::Vector3d boxSizeVector(boxSize, boxSize, boxSize);
		// std::cout<<valueGradient.transpose()<<std::endl;

	for(int i=0;i<valueSize;i++)
	{
		glPushMatrix();
		Eigen::Vector3d color = degreeToRgb(value[i]/1.0, true);
		glColor3f(color[0], color[1], color[2]);
		glTranslated(boxSize*i, 0, 0);
		drawCube(boxSizeVector);
		glPopMatrix();
	}
}
void 
GUI::drawSoccerLine(double x, double y)
{
	glBegin(GL_LINES);
	double floorDepth = -0.09;
	glVertex3f(0.0, y/2.0, floorDepth);
	glVertex3f(0.0, -y/2.0, floorDepth);
	glEnd();

	glBegin(GL_LINE_STRIP);
	glVertex3f(x/2.0, 20.15/15.0, floorDepth);
	glVertex3f(x/2.0-16.5/15.0, 20.15/15.0, floorDepth);
	glVertex3f(x/2.0-16.5/15.0, -20.15/15.0, floorDepth);
	glVertex3f(x/2.0, -20.15/15.0, floorDepth);
	glEnd();

	glBegin(GL_LINE_STRIP);
	glVertex3f(-x/2.0, 20.15/15.0, floorDepth);
	glVertex3f(-x/2.0+16.5/15.0, 20.15/15.0, floorDepth);
	glVertex3f(-x/2.0+16.5/15.0, -20.15/15.0, floorDepth);
	glVertex3f(-x/2.0, -20.15/15.0, floorDepth);
	glEnd();


	//penalty area


	glEnd();

	double radius = 9.15/15.0;

	glBegin(GL_LINE_LOOP);
	for(double i=0;i<360;i+=10.0)
	{
		double circleX = radius * cos(i/360 * 2 * M_PI);
		double circleY = radius * sin(i/360 * 2 * M_PI);
		glVertex3d(circleX, circleY, floorDepth);
	}
	glEnd();
}
