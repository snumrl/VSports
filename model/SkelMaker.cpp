#include "SkelMaker.h"
#include <dart/dart.hpp>
#include <iostream>

using namespace SkelMaker;
using namespace dart;
using namespace dart::dynamics;

dart::dynamics::BodyNode* 
SkelMaker::
makeFreeJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const SHAPE_TYPE& shapeType,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& pb2j,
	const Eigen::Isometry3d& cb2j
	)
{
	ShapePtr shape;
	if(shapeType == SHAPE_TYPE::BOX)
		shape = std::shared_ptr<BoxShape>(new BoxShape(size));
	else if(shapeType == SHAPE_TYPE::BALL)
		shape = std::shared_ptr<SphereShape>(new SphereShape(size[0]));
	
	BodyNode* bn;
	FreeJoint::Properties props;
	props.mName = body_name;

	props.mT_ParentBodyToJoint = pb2j;
	// props.mT_ChildBodyToJoint = cb2j;

	bn = target_skel->createJointAndBodyNodePair<FreeJoint>(
		parent, props, BodyNode::AspectProperties(body_name)).second;

	bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);

	return bn;
}

dart::dynamics::BodyNode* 
SkelMaker::
makeFree2DJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const SHAPE_TYPE& shapeType,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& pb2j,
	const Eigen::Isometry3d& cb2j
	)
{
	ShapePtr shape;
	if(shapeType == SHAPE_TYPE::BOX)
		shape = std::shared_ptr<BoxShape>(new BoxShape(size));
	else if(shapeType == SHAPE_TYPE::BALL)
		shape = std::shared_ptr<SphereShape>(new SphereShape(size[0]));
	else if(shapeType == SHAPE_TYPE::CYLINDER)
		shape = std::shared_ptr<CylinderShape>(new CylinderShape(size[0], size[1]));
	
	BodyNode* bn;
	TranslationalJoint2D::Properties props;
	props.mName = body_name;

	props.mT_ParentBodyToJoint = pb2j;
	// props.mT_ChildBodyToJoint = cb2j;

	bn = target_skel->createJointAndBodyNodePair<TranslationalJoint2D>(
		parent, props, BodyNode::AspectProperties(body_name)).second;

	bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);

	return bn;
}

dart::dynamics::BodyNode* 
SkelMaker::
makeWeldJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const SHAPE_TYPE& shapeType,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& pb2j,
	const Eigen::Isometry3d& cb2j
	)
{
	ShapePtr shape;
	if(shapeType == SHAPE_TYPE::BOX)
		shape = std::shared_ptr<BoxShape>(new BoxShape(size));
	else if(shapeType == SHAPE_TYPE::BALL)
		shape = std::shared_ptr<SphereShape>(new SphereShape(size[0]));

	BodyNode* bn;
	WeldJoint::Properties props;
	props.mName = body_name;

	props.mT_ParentBodyToJoint = pb2j;
	props.mT_ChildBodyToJoint = cb2j;

	bn = target_skel->createJointAndBodyNodePair<WeldJoint>(
		parent, props, BodyNode::AspectProperties(body_name)).second;

	bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);

	return bn;
}


dart::dynamics::BodyNode* 
SkelMaker::
makeRevoluteJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const SHAPE_TYPE& shapeType,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& pb2j,
	const Eigen::Isometry3d& cb2j
	)
{
	ShapePtr shape;
	if(shapeType == SHAPE_TYPE::BOX)
		shape = std::shared_ptr<BoxShape>(new BoxShape(size));
	else if(shapeType == SHAPE_TYPE::BALL)
		shape = std::shared_ptr<SphereShape>(new SphereShape(size[0]));
	
	BodyNode* bn;
	RevoluteJoint::Properties props;
	props.mName = body_name;

	props.mT_ParentBodyToJoint = pb2j;
	props.mT_ChildBodyToJoint = cb2j;

	bn = target_skel->createJointAndBodyNodePair<RevoluteJoint>(
		parent, props, BodyNode::AspectProperties(body_name)).second;

	bn->createShapeNodeWith<VisualAspect,  DynamicsAspect>(shape);

	return bn;
}


dart::dynamics::BodyNode* 
SkelMaker::
makeBallJointBody(
	const std::string& body_name,
	const dart::dynamics::SkeletonPtr& target_skel,
	dart::dynamics::BodyNode* const parent,
	const SHAPE_TYPE& shapeType,
	const Eigen::Vector3d& size,
	const Eigen::Isometry3d& pb2j,
	const Eigen::Isometry3d& cb2j
	)
{
	ShapePtr shape;
	if(shapeType == SHAPE_TYPE::BOX)
		shape = std::shared_ptr<BoxShape>(new BoxShape(size));
	else if(shapeType == SHAPE_TYPE::BALL)
		shape = std::shared_ptr<SphereShape>(new SphereShape(size[0]));
	
	BodyNode* bn;
	BallJoint::Properties props;
	props.mName = body_name;

	props.mT_ParentBodyToJoint = pb2j;
	props.mT_ChildBodyToJoint = cb2j;

	bn = target_skel->createJointAndBodyNodePair<BallJoint>(
		parent, props, BodyNode::AspectProperties(body_name)).second;

	bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);

	return bn;
}