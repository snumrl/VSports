// #include "vsports/SimpleSocWindow.h"
// #include "vsports/MultiSocWindow.h"
#include "vsports/InteractiveWindow.h"
#include "vsports/AgentWindow.h"
#include <GL/glut.h>
// #include <GL/glew.h>
// #include <GLFW/glfw3.h>
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

	IntWindow* simwindow;
	if(argc == 1)
	{
		simwindow = new IntWindow();
	}
	else if(argc==2)
		simwindow = new IntWindow(argv[1]);
	// else if(argc==3)
	// 	simwindow = new MultiSocWindow(argv);


	// glfwInit(&argc, argv);
	simwindow->initWindow(1000, 1000, "Render");
	AgentWindow* agentWindow = new AgentWindow(0,simwindow->mEnv);
	agentWindow->initWindow(96, 72, "Render");

	// simwindow->initWindow(1000, 1000, "Render2");
	simwindow->initialize();
	glutMainLoop();
}