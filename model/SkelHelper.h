#include <dart/dart.hpp>
namespace SkelHelper
{
dart::dynamics::SkeletonPtr makeFloor(double floorDepth = -0.1);
dart::dynamics::SkeletonPtr makeBall(double floorDepth = -0.1);
dart::dynamics::SkeletonPtr makeWall(double floorDepth = -0.1);
dart::dynamics::SkeletonPtr makeGoalpost(Eigen::Vector3d position, std::string label);
}