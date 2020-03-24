#include "vsports/BvhWindow.h"
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
			BvhWindow* simwindow = new BvhWindow(argv[2], argv[3]);
			simwindow->initWindow(1000, 1000, "Render");
			// simwindow->initialize();
		}
	}
	glutMainLoop();
}