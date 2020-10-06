#ifndef __SINGLECONTROL_WINDOW_H__
#define __SINGLECONTROL_WINDOW_H__
#include "../render/SimWindow.h"
#include "../sim/Character2D.h"
#include "../sim/Environment.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "common.h"
#include "../motion/BVHparser.h"
#include "../extern/ICA/plugin/MotionGenerator.h"
#include "../pyvs/Normalizer.h"


// #include <GL/glut.h>
// #include <GL/glew.h>
class SingleControlWindow : public SimWindow{
public:
	SingleControlWindow();
	SingleControlWindow(const char* nn_path, const char* control_nn_path);
	void initWindow(int _w, int _h, char* _name) override;

	void keyboard(unsigned char key, int x, int y) override;
	void keyboardUp(unsigned char key, int x, int y) override;
	void timer(int value) override;
	void mouse(int button, int state, int x, int y) override;
	void motion(int x, int y) override;
	int step();

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

    // void setBallPosition(bool leftContact);
    // void setBallVelocity(bool leftContact);

    // void updateHandTransform();

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
	boost::python::object *nn_module_0;
	boost::python::object *nn_module_1;
	boost::python::object *nn_module_2;

	boost::python::object *nn_module_decoders;

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
    // ICA::dart::MotionGenerator* mMotionGenerator;
	ICA::dart::MotionGeneratorBatch* mMotionGeneratorBatch;
	Eigen::VectorXd targetLocal;

    std::vector<std::vector<Eigen::Isometry3d>> prevHandTransforms;

    std::vector<std::vector<std::vector<double>>> xData;

    int mFrame;

    bool mTrackCharacter;

	std::vector<Eigen::VectorXd> mStates;

	Normalizer* mNormalizer;

	bool reducedDim;

	Eigen::VectorXd toOneHotVector(Eigen::VectorXd action);
	Eigen::VectorXd toOneHotVectorWithConstraint(int index, Eigen::VectorXd action);

	void showAvailableActions();


	void getControlMeanStdByActionType(int actionType);

	int getActionTypeFromVec(Eigen::VectorXd action);

	double fingerAngle;
	double fingerBallAngle;

	int latentSize;

    // Eigen::Vector3d curBallPosition;
    // Eigen::Vector3d prevBallPosition;
    // Eigen::Vector3d pprevBallPosition;
    // Eigen::Vector3d ppprevBallPosition;


};

#endif