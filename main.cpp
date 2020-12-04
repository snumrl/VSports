// #include "vsports/BvhWindow.h"
// #include "vsports/SingleBasketballWindow.h"
// #include "vsports/SingleControlWindow.h"
#include "vsports/MultiHeadWindow.h"
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

	if(argc==3)
	{

		// SingleControlWindow* simwindow = new SingleControlWindow(argv[1], argv[2]);
		MultiHeadWindow* simwindow = new MultiHeadWindow(argv[1], argv[2]);
		simwindow->initWindow(1500, 1200, "Render");
	}
	glutMainLoop();
}