#ifndef __GUI_GLUT_WINDOW_H__
#define __GUI_GLUT_WINDOW_H__
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <memory>

namespace GUI
{
class Camera;
class GLUTWindow
{
public:
	GLUTWindow();
	~GLUTWindow();

	virtual void initWindow(int _w, int _h, char* _name);

	static GLUTWindow* current();
	static void displayEvent();
	static void keyboardEvent(unsigned char key, int x, int y);
	static void keyboardUpEvent(unsigned char key, int x, int y);
	static void mouseEvent(int button, int state, int x, int y);
	static void motionEvent(int x, int y);
	static void reshapeEvent(int w, int h);
	static void timerEvent(int value);

	static std::vector<GLUTWindow*> mWindows;
	static std::vector<int> mWinIDs;

protected:
	virtual void initLights();
	virtual void display() = 0;
	virtual void keyboard(unsigned char key, int x, int y) = 0;
	virtual void keyboardUp(unsigned char key, int x, int y) = 0;
	virtual void mouse(int button, int state, int x, int y) = 0;
	virtual void motion(int x, int y) = 0;
	virtual void reshape(int w, int h) = 0;
	virtual void timer(int value) = 0;

	std::unique_ptr<Camera>				mCamera;
	bool 								mIsDrag;
	int 								mMouseType;
	int 								mPrevX, mPrevY;
	int 								mDisplayTimeout;

	std::vector<unsigned char> mScreenshotTemp;
	std::vector<unsigned char> mScreenshotTemp2;
};
}

#endif