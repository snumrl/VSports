#ifndef __BVH_WINDOW_H__
#define __BVH_WINDOW_H__
#include "../render/SimWindow.h"
#include "../sim/Character2D.h"
#include "../sim/Environment.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "common.h"
#include "../motion/BVHparser.h"
#include "../extern/ICA/plugin/MotionGenerator.h"

// #include <GL/glut.h>
// #include <GL/glew.h>
class BvhWindow : public SimWindow{
public:
	BvhWindow();
	BvhWindow(const char* bvh_path, const char* nn_path);
	void initWindow(int _w, int _h, char* _name) override;

	void keyboard(unsigned char key, int x, int y) override;
	void keyboardUp(unsigned char key, int x, int y) override;
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

	void initialize();


	void getActionFromNN(int index);

	void updateActionNoise(int index);

	void getValueFromNN(int index);

	void drawValueGradient();
	void drawValue();

	double getValue(int index);

	double getRNDFeatureDiff(int index);

	std::string indexToStateString(int index);

	Eigen::VectorXd getValueGradient(int index);

	double vsHardcodedAI_difficulty;

	dart::dynamics::SkeletonPtr makeGoalpost(Eigen::Vector3d position, std::string label);

	void initDartNameIdMapping();

	void applyKeyBoardEvent();

	void applyMouseEvent();


	dart::dynamics::SkeletonPtr floorSkel;
	dart::dynamics::SkeletonPtr ballSkel;

	dart::dynamics::SkeletonPtr redGoalpostSkel;
	dart::dynamics::SkeletonPtr blueGoalpostSkel;

	dart::dynamics::SkeletonPtr wallSkel;

	std::vector<Character2D*> charsRed;
	std::vector<Character2D*> charsBlue;

	Environment* mEnv;

	std::vector<Eigen::VectorXd> mSubgoalStates;
	std::vector<Eigen::VectorXd> mWSubgoalStates;
	std::vector<Eigen::VectorXd> mActions;

	std::vector<Eigen::VectorXd> mActionNoises;

	boost::python::object mm,mns,sys_module;
	boost::python::object *nn_module;

	// boost::python::object mm,mns,sys_module;
	boost::python::object target_rnd_nn_module;
	boost::python::object predictor_rnd_nn_module;
	// boost::python::object *reset_hidden;
	bool mIsNNLoaded;

	std::map<std::string, int> dartNameIdMap;

	bool controlOn;

	std::vector<Character2D*> mSubgoalCharacters;

	unsigned int programID;

	unsigned int vertexbuffer;

	bool vsHardcoded;

	BVHparser* bvhParser;

	int bvhFrame = 0;

	std::vector<std::string> charNames;

	bool showCourtMesh;
	int actionCount;

    int mPrevX;
    int mPrevY;

	int targetActionType;
	int actionDelay;

    Eigen::VectorXd goal;
    ICA::dart::MotionGenerator* mMotionGenerator;

	Eigen::VectorXd targetLocal;
};

#endif