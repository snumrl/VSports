#ifndef __BEHAVIOR_TREE_H__
#define __BEHAVIOR_TREE_H__
#include <Eigen/Dense>
#include <vector>
#include <iostream>
enum BNType {ROOT, SEQUENCE, SELECTOR, EXECUTION, IF, WHILE};
enum resultType {SUCCESS, FAILURE, RUNNING};
class BNode
{
public:
	BNode(std::string name, BNType type, BNode* parentNode = nullptr);

	std::string getName(){return name;}
	BNType getType(){return type;}
	BNode* getParentNode(){return parentNode;}
	Eigen::VectorXd getAction(Eigen::VectorXd curState);
	void addChildNode(BNode* childNode);
	resultType step(Eigen::VectorXd curState);

	void setActionFunction(Eigen::VectorXd (*actionFunc)(Eigen::VectorXd));
	void setConditionFunction(bool (*conditionFunc)(Eigen::VectorXd));

	Eigen::VectorXd getActionFromBTree(Eigen::VectorXd curState);


	std::string name;
	BNType type;
	BNode* parentNode;
	std::vector<BNode*> childNodes;
	int curChildNodeIndex;	//for sequence bnode
	Eigen::VectorXd (*actionFunc)(Eigen::VectorXd);
	bool (*conditionFunc)(Eigen::VectorXd);
};


#endif