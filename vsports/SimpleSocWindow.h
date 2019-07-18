#ifndef __SIMPLE_SOC_WINDOW_H__
#define __SIMPLE_SOC_WINDOW_H__
#include "../render/SimWindow.h"
#include "../sim/Character2D.h"
#include "../sim/Environment.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

class SimpleSocWindow : public SimWindow{
public:
	SimpleSocWindow();
	SimpleSocWindow(const std::string& nn_path);

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

	void getActionFromNN(bool vsHardcodedAI = false);
	double vsHardcodedAI_difficulty;

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

	boost::python::object mm,mns,sys_module,nn_module,muscle_nn_module;
	bool mIsNNLoaded;
};

#endif