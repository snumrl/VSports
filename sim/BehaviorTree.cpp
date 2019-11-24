#include "BehaviorTree.h"
#include <iostream>

using namespace std;

BNode::
BNode(std::string name, BNType type, BNode* parentNode)
: name(name), type(type), parentNode(parentNode)
{
	this->actionFunc = nullptr;
	if(parentNode != nullptr)
		parentNode->addChildNode(this);
	curChildNodeIndex = 0;
}

void
BNode::
addChildNode(BNode* childNode)
{
	childNodes.push_back(childNode);
	childNode->parentNode = this;
}

void 
BNode::
setActionFunction(Eigen::VectorXd (*actionFunc)(Eigen::VectorXd))
{
	this->actionFunc = actionFunc;
}

void 
BNode::
setConditionFunction(bool (*conditionFunc)(Eigen::VectorXd))
{
	this->conditionFunc = conditionFunc;
}


// Should be follower after step
Eigen::VectorXd 
BNode::
getAction(Eigen::VectorXd curState)
{
	if(type == BNType::EXECUTION)
	{
		assert(actionFunc != nullptr);
		// cout<<name<<endl;
		return actionFunc(curState);
	}
	return childNodes[curChildNodeIndex]->getAction(curState);
}

resultType
BNode::
step(Eigen::VectorXd curState)
{
	resultType result;
		// cout<<name<<endl;

	// every execution action are always possible.
	if(type == BNType::EXECUTION)
	{
		// cout<<name<<endl;
		return resultType::SUCCESS;
	}

	

	resultType childResult = childNodes[curChildNodeIndex]->step(curState);

	if(childResult == resultType::FAILURE)
	{
		switch(type)
		{
			// case BNType::WHILE:
			// 	result = resultType::RUNNING;
			// 	break;
			case BNType::SEQUENCE:
				curChildNodeIndex = 0;
				result = resultType::FAILURE;
				break;
			default:
				result = resultType::FAILURE;
				break;
		}
	}
	else if(childResult == resultType::SUCCESS)
	{
		switch(type)
		{
			// case BNType::WHILE:
			// 	result = resultType::SUCCESS;
			// 	break;
			case BNType::SEQUENCE:
				curChildNodeIndex++;
				// cout<<curChildNodeIndex<<endl;
				if(curChildNodeIndex >= childNodes.size())
				{
					curChildNodeIndex = 0;
					result = resultType::SUCCESS;
				}
				else
					result = resultType::RUNNING;
			break;

			default:
				result = resultType::SUCCESS;
				break;
		}
	}
	else
	{
		result = resultType::RUNNING;
	}

	if(type == BNType::WHILE)
	{
		// cout<<name<<" "<<conditionFunc(curState)<<endl;
		// cout<<(curState.segment(4,2).norm() >= 0.05)<<endl;
		if(conditionFunc(curState))
		{
			return resultType::RUNNING;
		}
	}
	if(type == BNType::IF)
	{
		if(!conditionFunc(curState))
		{
			return resultType::FAILURE;
		}
	}

	return result;
}

Eigen::VectorXd
BNode::
getActionFromBTree(Eigen::VectorXd curState)
{
	assert(type == BNType::ROOT);
	step(curState);
	return getAction(curState);
}