#include "vsports/SimpleSocWindow.h"
#include <GL/glut.h>
#include <iostream>
#include <stdio.h>
using namespace std;

namespace p = boost::python;
namespace np = boost::python::numpy;

int main(int argc, char** argv)
{
	Py_Initialize();
	np::initialize();



	SimpleSocWindow* simwindow;
	if(argc == 1)
		simwindow = new SimpleSocWindow();
	else if(argc==2)
		simwindow = new SimpleSocWindow(argv[1]);


	glutInit(&argc, argv);
	simwindow->initWindow(1000, 1000, "Render");
	glutMainLoop();
}