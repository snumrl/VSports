#include "vsports/SimpleSocWindow.h"
#include <GL/glut.h>
#include <iostream>
#include <stdio.h>
using namespace std;

int main(int argc, char** argv)
{

	SimpleSocWindow* simwindow = new SimpleSocWindow();


	glutInit(&argc, argv);
	simwindow->initWindow(1000, 1000, "Render");
	glutMainLoop();
}