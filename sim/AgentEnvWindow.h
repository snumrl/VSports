#ifndef __AGENT_ENV_WINDOW_H__
#define __AGENT_ENV_WINDOW_H__
#include "../render/SimWindow.h"

class Environment;

class AgentEnvWindow : public SimWindow{
public:
	AgentEnvWindow(int index, Environment* env);
	void initWindow(int _w, int _h, char* _name) override;

	void keyboard(unsigned char key, int x, int y) override;
	void keyboardUp(unsigned char key, int x, int y) override;
	void timer(int value) override;
	void mouse(int button, int state, int x, int y) override;
	void motion(int x, int y) override;


	void display() override;
	void initGoalpost();

	void setAgentView();

	void getAgentView();

	dart::dynamics::SkeletonPtr makeGoalpost(Eigen::Vector3d position, std::string label);

	dart::dynamics::SkeletonPtr floorSkel;
	dart::dynamics::SkeletonPtr ballSkel;

	dart::dynamics::SkeletonPtr redGoalpostSkel;
	dart::dynamics::SkeletonPtr blueGoalpostSkel;

	dart::dynamics::SkeletonPtr wallSkel;

	Environment* mEnv;

	int mIndex;

	std::vector<unsigned char> agentViewImgTemp;
	std::vector<unsigned char> agentViewImg;

	double floorDepth = -0.1;
};
#endif