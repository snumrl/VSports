#include <dart/dart.hpp>
namespace SkelHelper
{
dart::dynamics::SkeletonPtr makeFloor(double floorDepth = -0.0);
dart::dynamics::SkeletonPtr makeBall(double floorDepth = -0.0);
dart::dynamics::SkeletonPtr makeWall(double floorDepth = -0.0);
dart::dynamics::SkeletonPtr makeBasketBallFloor(double floorDepth = -0.0);
dart::dynamics::SkeletonPtr makeSimpleFloor(double floorDepth = -0.0);
dart::dynamics::SkeletonPtr makeGoalpost(Eigen::Vector3d position, std::string label);
dart::dynamics::BodyNode* MakeWeldJointBody(
		const std::string& body_name,
		const std::string& obj_file,
		const dart::dynamics::SkeletonPtr& target_skel,
		dart::dynamics::BodyNode* const parent,
		const Eigen::Vector3d& size,
		const Eigen::Isometry3d& joint_position,
		const Eigen::Isometry3d& body_position,
		double mass,
		bool contact,
		bool obj_visual_con);
}