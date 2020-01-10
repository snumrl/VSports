#ifndef __EPISODE_WINDOW_H__
#define __EPISODE_WINDOW_H__
#include "../render/SimWindow.h"
#include "../sim/Character2D.h"
#include "../sim/Environment.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
// #include <GL/glut.h>
// #include <GL/glew.h>
class EpiWindow : public SimWindow{
public:
	EpiWindow();
	EpiWindow(const std::string& replay_path, const std::string& numPath);
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

	void applyKeyEvent();

	void getActionFromNN(int index);
	void getValueFromNN(int index);

	void drawValueGradient();
	void drawValue();

	double getValue(int index);

	std::string indexToStateString(int index);

	Eigen::VectorXd getValueGradient(int index);

	double vsHardcodedAI_difficulty;

	dart::dynamics::SkeletonPtr makeGoalpost(Eigen::Vector3d position, std::string label);

	void storeEpisodeFromPath();
	void reconEnvFromCurrentState();

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


	bool controlOn;

	std::vector<Character2D*> mSubgoalCharacters;

	unsigned int programID;

	unsigned int vertexbuffer;

	int mNumPath;
	double floorDepth = -0.1;

	std::vector<std::string> mReplayPathList;

	int curPathIndex;
	int curTimeStep;

	std::vector<std::vector<Eigen::VectorXd>> stateList;
	std::vector<std::vector<double>> TDList;
	std::vector<int> charIndexList;

	double curTD;


};

#endif