#ifndef __SIM_WINDOW__
#define __SIM_WINDOW__
#include "Camera.h"
#include "GLUTWindow.h"
#include <string>
#include <dart/dart.hpp>

class SimWindow : public GUI::GLUTWindow
{
public:
	SimWindow();
	dart::simulation::WorldPtr mWorld;

protected:
	/// Draw all the skeletons in mWorld. Lights and Camera are operated here.
	void display() override;

	/// The user interactions with keyboard.
	void keyboard(unsigned char key, int x, int y) override;

	/// Stores the data for SimWindow::Motion.
	void mouse(int button, int state, int x, int y) override;

	/// The user interactions with mouse. Camera view is set here.
	void motion(int x, int y) override;

	/// Reaction to window resizing.
	void reshape(int w, int h) override;

	/// timer
	void timer(int value) override;

	/// Screenshot. The png file will be stored as ./frames/Capture/[number].png
	void screenshot();

	// /// Set the skeleton positions in mWorld to the positions at n frame.
	// void setFrame(int n);

	// /// Set the skeleton positions in mWorld to the positions at the next frame.
	// void nextFrame();

	// /// Set the skeleton positions in mWorld to the positions at 1/30 sec lator.
	// void nextFrameRealTime();

	// /// Set the skeleton positions in mWorld to the positions at the previous frame
	// void prevFrame();

	double mTimeStep;
	bool mIsDrag;
	int mMouseType;
};

#endif