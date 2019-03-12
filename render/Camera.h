#ifndef __GUI_CAMAERA_H__
#define __GUI_CAMAERA_H__
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

namespace GUI
{
class Camera
{
public:
	Camera();

	void setCamera(const Eigen::Vector3d& lookAt, const Eigen::Vector3d& eye, const Eigen::Vector3d& up);
	void apply();

	void pan(int x, int y, int prev_x, int prev_y);
	void zoom(int x, int y, int prev_x, int prev_y);
	void rotate(int x, int y, int prev_x, int prev_y);
	void translate(int x, int y, int prev_x, int prev_y);

	void setCenter(const Eigen::Vector3d& c);

	void setLookAt(const Eigen::Vector3d& lookAt);

	Eigen::Vector3d rotateq(const Eigen::Vector3d& target, const Eigen::Vector3d& rotateVector, double angle);
	Eigen::Vector3d getTrackballPoint(int mouseX, int mouseY, int w, int h);


	Eigen::Vector3d lookAt;
	Eigen::Vector3d eye;
	Eigen::Vector3d up;
	double fovy;

};
}

#endif