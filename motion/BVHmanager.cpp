#include "BVHmanager.h"
#include "BVHparser.h"
#include <dart/dart.hpp>

void BVHmanager::setPositionFromBVH(dart::dynamics::SkeletonPtr skel, BVHparser *bvhParser, int bvh_frame)
{
	MotionNode* rootNode = bvhParser->getRootNode();
	MotionNode* curNode  = rootNode->getNextNode();

	skel->setPosition(skel->getDof("j_"+rootNode->getName()+"_pos_x")->getIndexInSkeleton(), rootNode->data[bvh_frame][0]/100.0);
	skel->setPosition(skel->getDof("j_"+rootNode->getName()+"_pos_y")->getIndexInSkeleton(), rootNode->data[bvh_frame][1]/100.0);
	skel->setPosition(skel->getDof("j_"+rootNode->getName()+"_pos_z")->getIndexInSkeleton(), rootNode->data[bvh_frame][2]/100.0);

	skel->setPosition(skel->getDof("j_"+rootNode->getName()+"_rot_x")->getIndexInSkeleton(), rootNode->data[bvh_frame][3]);
	skel->setPosition(skel->getDof("j_"+rootNode->getName()+"_rot_y")->getIndexInSkeleton(), rootNode->data[bvh_frame][4]);
	skel->setPosition(skel->getDof("j_"+rootNode->getName()+"_rot_z")->getIndexInSkeleton(), rootNode->data[bvh_frame][5]);
	//skel->getJoint(rootNode->getName())->setPositions()

	
	while(curNode != nullptr)
	{
		skel->setPosition(skel->getDof("j_"+curNode->getName()+"_x")->getIndexInSkeleton(), curNode->data[bvh_frame][0]);
		skel->setPosition(skel->getDof("j_"+curNode->getName()+"_y")->getIndexInSkeleton(), curNode->data[bvh_frame][1]);
		skel->setPosition(skel->getDof("j_"+curNode->getName()+"_z")->getIndexInSkeleton(), curNode->data[bvh_frame][2]);
		curNode = curNode->getNextNode();
	}

}