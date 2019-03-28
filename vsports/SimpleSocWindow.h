#ifndef __SIMPLE_SOC_WINDOW_H__
#define __SIMPLE_SOC_WINDOW_H__
#include "../render/SimWindow.h"
#include "../sim/Character2D.h"

class SimpleSocWindow : public SimWindow{
public:
	SimpleSocWindow();

	void keyboard(unsigned char key, int x, int y) override;
	void timer(int value) override;
	void mouse(int button, int state, int x, int y) override;
	void motion(int x, int y) override;

	void display() override;

	void initFloor();
	void initCharacters();
	void initBall();
	void initCustomView();

	dart::dynamics::SkeletonPtr makeFloor();
	dart::dynamics::SkeletonPtr makeBall();

	dart::dynamics::SkeletonPtr floorSkel;
	dart::dynamics::SkeletonPtr ballSkel;

	std::vector<Character2D*> charsRed;
	std::vector<Character2D*> charsBlue;
};

#endif