// #include "vsports/SimpleSocWindow.h"
// #include "vsports/MultiSocWindow.h"
#include "vsports/InteractiveWindow.h"
#include "vsports/ImitationWindow.h"
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

	// SimWindow* simwindow;
	if(argc == 1)
	{
		ImitationWindow* simwindow = new ImitationWindow();
		simwindow->initWindow(1000, 1000, "Render");
		simwindow->initialize();
	}
	else if(argc==3)
	{
		IntWindow* simwindow = new IntWindow(argv[1], argv[2]);
		simwindow->initWindow(1000, 1000, "Render");
		simwindow->initialize();
	}

	else if(argc==5)
	{
		IntWindow* simwindow = new IntWindow(argv[1], argv[2], argv[3], argv[4]);
		simwindow->initWindow(1000, 1000, "Render");
		simwindow->initialize();
	}
	else if(argc == 4)
	{
		ImitationWindow* simwindow = new ImitationWindow(argv[1], argv[2]);
		simwindow->initWindow(1000, 1000, "Render");
		simwindow->initialize();
	}


	// else if(argc==3)
	// 	simwindow = new MultiSocWindow(argv);


	// glfwInit(&argc, argv);
	// AgentWindow* agentWindow = new AgentWindow(0,simwindow->mEnv);
	// agentWindow->initWindow(400, 300, "Render");

	// simwindow->initWindow(1000, 1000, "Render2");

	glutMainLoop();
}