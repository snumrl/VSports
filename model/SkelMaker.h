#ifndef __MODEL_SKELMAKER_H__
#define __MODEL_SKELMAKER_H__
#include <dart/dart.hpp>

enum JOINT_TYPE
{
	WELD,
	REVOLUTE,
	UNIVERSAL,
	EULER,
	BALL_AND_SOCKET,
	FREE_2D,
	FREE
};

enum SHAPE_TYPE
{
	BOX,
	BALL,
	CYLINDER
};

namespace SkelMaker
{
dart::dynamics::BodyNode* makeFreeJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const SHAPE_TYPE& shapeType,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& pb2j,
	const Eigen::Isometry3d& cb2j
	);

dart::dynamics::BodyNode* makeFree2DJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const SHAPE_TYPE& shapeType,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& pb2j,
	const Eigen::Isometry3d& cb2j
	);

dart::dynamics::BodyNode* makeWeldJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const SHAPE_TYPE& shapeType,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& pb2j,
	const Eigen::Isometry3d& cb2j
	);

dart::dynamics::BodyNode* makeRevoluteJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const SHAPE_TYPE& shapeType,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& pb2j,
	const Eigen::Isometry3d& cb2j
	);

dart::dynamics::BodyNode* makeBallJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const SHAPE_TYPE& shapeType,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& pb2j,
	const Eigen::Isometry3d& cb2j
	);

}


#endif