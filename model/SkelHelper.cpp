#include "SkelHelper.h"
#include "SkelMaker.h"
using namespace dart;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace std;

SkeletonPtr
SkelHelper::
makeFloor(double floorDepth)
{
	SkeletonPtr floor = Skeleton::create("floor");
	Eigen::Vector3d position = Eigen::Vector3d(0.0, 0.0, floorDepth);
	Eigen::Isometry3d cb2j;
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"floor", floor, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(14.0, 0.01, 15.0),
	Eigen::Isometry3d::Identity(), cb2j);
	return floor;
}

SkeletonPtr 
SkelHelper::
makeBall(double floorDepth)
{
	SkeletonPtr ball = Skeleton::create("ball");
	Eigen::Vector3d position = Eigen::Vector3d(0.0, 0.0, floorDepth + 0.08);
	Eigen::Isometry3d cb2j;
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeFreeJointBody(
	"ball", ball, nullptr,
	SHAPE_TYPE::BALL,
	Eigen::Vector3d::UnitX()*0.12,
	Eigen::Isometry3d::Identity(), cb2j);
	return ball;
}

SkeletonPtr
SkelHelper::
makeWall(double floorDepth)
{
	double goalpostSize = 1.5;
	double goalLineSize = 6.0;
	SkeletonPtr wall = Skeleton::create("wall");
	Eigen::Vector3d position = Eigen::Vector3d(
		4.0, 
		goalpostSize/2.0 + (goalLineSize-goalpostSize)/4.0, 
		0.1 + floorDepth);
	Eigen::Isometry3d cb2j;
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall0_0", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(0.1, 2.25, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);


	position = Eigen::Vector3d(
		4.0, 
		-(goalpostSize/2.0 + (goalLineSize-goalpostSize)/4.0), 
		0.1 + floorDepth);
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall0_1", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(0.1, 2.25, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);

	position = Eigen::Vector3d(
		-4.0, 
		(goalpostSize/2.0 + (goalLineSize-goalpostSize)/4.0), 
		0.1 + floorDepth);
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall1_0", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(0.1, 2.25, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);

	position = Eigen::Vector3d(
		-4.0, 
		-(goalpostSize/2.0 + (goalLineSize-goalpostSize)/4.0), 
		0.1 + floorDepth);
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall1_1", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(0.1, 2.25, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);

	position = Eigen::Vector3d(0.0, 3.0, 0.1 + floorDepth);
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall2", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(8.0, 0.1, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);

	position = Eigen::Vector3d(0.0, -3.0, 0.1 + floorDepth);
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"wall3", wall, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(8.0, 0.1, 0.2),
	Eigen::Isometry3d::Identity(), cb2j);
	return wall;
}

SkeletonPtr 
SkelHelper::
makeGoalpost(Eigen::Vector3d position, std::string label)
{
	SkeletonPtr goalpost = Skeleton::create("goalpost_"+label);
	Eigen::Isometry3d cb2j;
	cb2j.setIdentity();
	cb2j.translation() += -position;
	SkelMaker::makeWeldJointBody(
	"goalpost_"+label, goalpost, nullptr,
	SHAPE_TYPE::BOX,
	Eigen::Vector3d(0.1, 1.5, 0.5),
	Eigen::Isometry3d::Identity(), cb2j);
	return goalpost;
}


SkeletonPtr
SkelHelper::
makeBasketBallFloor(double floorDepth)
{
	SkeletonPtr floor = Skeleton::create("floor");
	Eigen::Isometry3d pb2j;
	Eigen::Isometry3d cb2j;
	pb2j.setIdentity();
	cb2j.setIdentity();

	// Eigen::Matrix3d mat = Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitX());
	cb2j.linear() *=  Eigen::AngleAxisd(M_PI/2.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
	cb2j.linear() *=  Eigen::AngleAxisd(-M_PI/2.0, Eigen::Vector3d::UnitX()).toRotationMatrix();
	// cb2j.translation() = Eigen::Vector3d(-0.46, 0.0, 0.0);
	char resolved_path[PATH_MAX]; 
	realpath("../", resolved_path);
	std::string absolutePathProject = resolved_path;
	// std::cout<<absolutePathProject+"/data/models/BasketBallCourt_obj/basketballcourt.obj"<<std::endl;

	SkelHelper::MakeWeldJointBody("floor", 
						absolutePathProject+"/data/models/BasketBallCourt_obj/basketballcourt.obj",
						floor,
						nullptr,
						Eigen::Vector3d(15.0, 28.0, 0.01),
						pb2j,
						cb2j,
						1.0,
						true,
						true);

	// Eigen::Vector3d position = Eigen::Vector3d(0.0, 0.0, floorDepth);
	// Eigen::Isometry3d cb2j;
	// cb2j.setIdentity();
	// cb2j.translation() += -position;
	// SkelMaker::makeWeldJointBody(
	// "floor", floor, nullptr,
	// SHAPE_TYPE::BOX,
	// Eigen::Vector3d(14.0, 0.01, 15.0),
	// Eigen::Isometry3d::Identity(), cb2j);
	return floor;
}

BodyNode* 
SkelHelper::
MakeWeldJointBody(
		const std::string& body_name,
		const std::string& obj_file,
		const dart::dynamics::SkeletonPtr& target_skel,
		dart::dynamics::BodyNode* const parent,
		const Eigen::Vector3d& size,
		const Eigen::Isometry3d& joint_position,
		const Eigen::Isometry3d& body_position,
		double mass,
		bool contact,
		bool obj_visual_consensus)
{
	ShapePtr shape = std::shared_ptr<BoxShape>(new BoxShape(size));

	dart::dynamics::Inertia inertia;
	inertia.setMass(mass);
	inertia.setMoment(shape->computeInertia(mass));

	BodyNode* bn;
	FreeJoint::Properties props;
	props.mName = body_name;
	// props.mT_ChildBodyToJoint = joint_position;
	props.mT_ParentBodyToJoint = body_position;


	bn = target_skel->createJointAndBodyNodePair<WeldJoint>(
			parent,props,BodyNode::AspectProperties(body_name)).second;

	if(contact)
		bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
	else
		bn->createShapeNodeWith<VisualAspect, DynamicsAspect>(shape);
	if(obj_file!="None")
	{
		if (obj_visual_consensus){
		// cout<<"11111"<<endl;
			std::string obj_path = obj_file;
			const aiScene* mesh = MeshShape::loadMesh(std::string(obj_path));
		// cout<<"2222"<<endl;
			double scale = 7.5/385.0*1.0;
			ShapePtr visual_shape = std::shared_ptr<MeshShape>(new MeshShape(Eigen::Vector3d(scale,scale,scale),mesh));
			// bn->createShapeNodeWith<VisualAspect>(visual_shape);
			auto vsn = bn->createShapeNodeWith<VisualAspect>(visual_shape);
			Eigen::Isometry3d T_visual;
			T_visual.setIdentity();
			T_visual.translation() = Eigen::Vector3d(0.035, 0.58, 0.0);
			vsn->setRelativeTransform(T_visual);

		}else{
			std::string obj_path = obj_file;
			const aiScene* mesh = MeshShape::loadMesh(std::string(obj_path));
			ShapePtr visual_shape = std::shared_ptr<MeshShape>(new MeshShape(Eigen::Vector3d(0.01,0.01,0.01),mesh));
			auto vsn = bn->createShapeNodeWith<VisualAspect>(visual_shape);
			Eigen::Isometry3d T_visual;
			T_visual.setIdentity();
			T_visual = body_position.inverse();
			vsn->setRelativeTransform(T_visual);
		}
	}
	bn->setInertia(inertia);
	bn->getTransform();
	// bn->setRestitutionCoeff(0.0);
	return bn;
}
