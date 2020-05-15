// #include "vsports/BvhWindow.h"
#include "vsports/SingleBasketballWindow.h"
#include "vsports/SingleControlWindow.h"
#include "vsports/common.h"

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

	glutInit(&argc, argv);
	if(argc==4)
	{
		if(strcmp(argv[1], "-bvh")==0)
		{
			SingleBasketballWindow* simwindow = new SingleBasketballWindow(argv[2], argv[3]);
			simwindow->initWindow(1800, 1200, "Render");
			// simwindow->initialize();
		}
	}
	else if(argc==5)
	{
		if(strcmp(argv[1], "-bvh")==0)
		{
			SingleControlWindow* simwindow = new SingleControlWindow(argv[2], argv[3], argv[4]);
			simwindow->initWindow(1800, 1200, "Render");
			// simwindow->initialize();
		}
	}
	glutMainLoop();
}