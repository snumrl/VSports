#ifndef __MULTI_SOC_WINDOW_H__
#define __MULTI_SOC_WINDOW_H__
#include "../render/SimWindow.h"
#include "../sim/Character2D.h"
#include "../sim/Environment.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

class MultiSocWindow : public SimWindow{
public:
	MultiSocWindow();
	// MultiSocWindow(const std::string& nn_path0, const std::string& nn_path1);
	MultiSocWindow(char** paths);

	void keyboard(unsigned char key, int x, int y) override;
	void timer(int value) override;
	void mouse(int button, int state, int x, int y) override;
	void motion(int x, int y) override;
	void step();

	void display() override;

	void initFloor();
	void initCharacters();
	void initBall();
	void initGoalpost();
	void initCustomView();

	void getActionFromNN();

	dart::dynamics::SkeletonPtr makeGoalpost(Eigen::Vector3d position, std::string label);

	dart::dynamics::SkeletonPtr floorSkel;
	dart::dynamics::SkeletonPtr ballSkel;

	dart::dynamics::SkeletonPtr redGoalpostSkel;
	dart::dynamics::SkeletonPtr blueGoalpostSkel;

	dart::dynamics::SkeletonPtr wallSkel;

	std::vector<Character2D*> charsRed;
	std::vector<Character2D*> charsBlue;

	Environment* mEnv;

	std::vector<Eigen::VectorXd> mActions;

	boost::python::object mm,mns,sys_module;
	std::vector<boost::python::object> nn_modules;
	bool mIsNNLoaded;
	double floorDepth = -0.1;
};

#endif