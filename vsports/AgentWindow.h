// #ifndef __AGENT_WINDOW_H__
// #define __AGENT_WINDOW_H__
// #include "../render/SimWindow.h"
// #include "../sim/Character2D.h"
// #include "../sim/Environment.h"
// #include <boost/python.hpp>
// #include <boost/python/numpy.hpp>
// // #include <GL/glut.h>
// // #include <GL/glew.h>
// class AgentWindow : public SimWindow{
// public:
// 	AgentWindow(int index, Environment* env);
// 	AgentWindow(int index, Environment* env, const std::string& nn_path);
// 	void initWindow(int _w, int _h, char* _name) override;

// 	void keyboard(unsigned char key, int x, int y) override;
// 	void keyboardUp(unsigned char key, int x, int y) override;
// 	void timer(int value) override;
// 	void mouse(int button, int state, int x, int y) override;
// 	void motion(int x, int y) override;
// 	void step();

// 	void display() override;

// 	void initFloor();
// 	void initCharacters();
// 	void initBall();
// 	void initGoalpost();
// 	void initCustomView();

// 	void setAgentView();

// 	void initialize();

// 	void applyKeyEvent();

// 	void getActionFromNN(bool vsHardcodedAI = false);

// 	void drawValueGradient();

// 	std::string indexToStateString(int index);

// 	Eigen::VectorXd getValueGradient(int index);

// 	void getAgentView();

// 	double vsHardcodedAI_difficulty;

// 	dart::dynamics::SkeletonPtr makeGoalpost(Eigen::Vector3d position, std::string label);

// 	dart::dynamics::SkeletonPtr redGoalpostSkel;
// 	dart::dynamics::SkeletonPtr blueGoalpostSkel;

// 	std::vector<Character2D*> charsRed;
// 	std::vector<Character2D*> charsBlue;

// 	Environment* mEnv;

// 	std::vector<Eigen::VectorXd> mSubgoalStates;
// 	std::vector<Eigen::VectorXd> mWSubgoalStates;
// 	std::vector<Eigen::VectorXd> mActions;

// 	boost::python::object mm,mns,sys_module;
// 	boost::python::object *nn_sc_module;
// 	boost::python::object *nn_la_module;
// 	boost::python::object *reset_sc_hidden;
// 	boost::python::object *reset_la_hidden;
// 	bool mIsNNLoaded;

// 	bool controlOn;

// 	std::vector<Character2D*> mSubgoalCharacters;

// 	unsigned int programID;

// 	unsigned int vertexbuffer;

// 	int mIndex;

// 	std::vector<unsigned char> agentViewImgTemp;
// 	std::vector<unsigned char> agentViewImg;
// };

// #endif