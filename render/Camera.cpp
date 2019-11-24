#include "Camera.h"
#include <GL/glut.h>
#include <iostream>
using namespace GUI;

Camera::
Camera()
:fovy(60.0), lookAt(Eigen::Vector3d(0.0, 0.0, 0.0)), eye(Eigen::Vector3d(0.0, 0.0, 3.0)), up(Eigen::Vector3d(0.0, 1.0, 0.0))
{

}

void
Camera::
setCamera(const Eigen::Vector3d& lookAt, const Eigen::Vector3d& eye, const Eigen::Vector3d& up)
{
	this->lookAt = lookAt;
	this->eye = eye;
	this->up = up;
}

void
Camera::
apply()
{
	GLint w = glutGet(GLUT_WINDOW_WIDTH);
	GLint h = glutGet(GLUT_WINDOW_HEIGHT);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	const double size = 1.5;
	gluPerspective(fovy, (GLfloat)w / (GLfloat)h, 0.01, 1000);
	// glOrtho(-size, size, -size, size, 0.01, 100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(eye.x(), eye.y(), eye.z(),
			lookAt.x(), lookAt.y(), lookAt.z(),
			up.x(), up.y(), up.z());
}

// void
// Camera::
// pan(int x, int y, int prev_x, int prev_y)
// {
// 	double delta =
// }

void
Camera::
zoom(int x, int y, int prev_x, int prev_y)
{
	double delta = (prev_y - y)/100.0;
	Eigen::Vector3d vec = (lookAt - eye);
	double scale = vec.norm();
	scale = std::max((scale-delta), 1.0);
	Eigen::Vector3d vd = (scale - delta) * (lookAt - eye).normalized();
	eye = lookAt - vd;
}

void
Camera::
rotate(int x,int y,int prev_x,int prev_y)
{
	// std::cout<<x<<" "<<prev_x<<std::endl;
	// std::cout<<y<<" "<<prev_y<<std::endl;
	GLint w = glutGet(GLUT_WINDOW_WIDTH);
	GLint h = glutGet(GLUT_WINDOW_HEIGHT);

	Eigen::Vector3d prevPoint = getTrackballPoint(prev_x,prev_y,w,h).normalized();
	Eigen::Vector3d curPoint = getTrackballPoint(x,y,w,h).normalized();
	Eigen::Vector3d rotVec = prevPoint.cross(curPoint);

	Eigen::Vector3d basisZ_Vec = (eye - lookAt).normalized();
	Eigen::Vector3d basisY_Vec = up;
	Eigen::Vector3d basisX_Vec = basisY_Vec.cross(basisZ_Vec).normalized();

	Eigen::Matrix3d rotMat;
	rotMat.block<3,1>(0,0) = basisX_Vec;
	rotMat.block<3,1>(0,1) = basisY_Vec;
	rotMat.block<3,1>(0,2) = basisZ_Vec;

	if(rotVec.norm() != 0)
		rotVec = rotMat * rotVec.normalized();


	// rotVec = UnProject(rotVec);
	double cosT = prevPoint.dot(curPoint);
	double sinT = (prevPoint.cross(curPoint)).norm();

	double angle = -atan2(sinT, cosT);

	this->eye = Eigen::AngleAxisd(angle, rotVec)*(this->eye - this->lookAt) + this->lookAt;
	this->up = Eigen::AngleAxisd(angle, rotVec)*this->up;

	// Eigen::Vector3d n = this->lookAt - this->eye;
	// n = rotateq(n, rotVec, angle);
	// this->up = rotateq(this->up, rotVec, angle);
	// this->eye = this->lookAt - n;
}

Eigen::Vector3d
Camera::
rotateq(const Eigen::Vector3d& target, const Eigen::Vector3d& rotateVector, double angle)
{
	Eigen::Vector3d rv = rotateVector.normalized();

	Eigen::Quaternion<double> rot(cos(angle / 2.0), sin(angle / 2.0)*rv.x(), sin(angle / 2.0)*rv.y(), sin(angle / 2.0)*rv.z());
	rot.normalize();
	Eigen::Quaternion<double> tar(0, target.x(), target.y(), target.z());


	tar = rot.inverse()*tar*rot;

	return Eigen::Vector3d(tar.x(), tar.y(), tar.z());
}

Eigen::Vector3d
Camera::
getTrackballPoint(int mouseX, int mouseY, int w, int h)
{
	double rad = sqrt((double)(w*w+h*h)) / 2.0;
	double dx = (double)(mouseX)-(double)w / 2.0;
	double dy = -1*((double)(mouseY)-(double)h / 2.0);
	double dx2pdy2 = dx*dx + dy*dy;

	if (rad*rad - dx2pdy2 <= 0)
		return Eigen::Vector3d(dx, dy, 0);
	else
		return Eigen::Vector3d(dx, dy, sqrt(rad*rad - dx*dx - dy*dy));
}

void
Camera::
setCenter(const Eigen::Vector3d& c){
	Eigen::Vector3d delta = c - lookAt;
	lookAt += delta; eye += delta;
}

void
Camera::
setLookAt(const Eigen::Vector3d& lookAt)
{
	this->lookAt = lookAt;
	this->eye = lookAt + Eigen::Vector3d(0.0, 0.0, 2.0);
}

void
Camera::
translate(int x, int y, int prev_x, int prev_y)
{
	Eigen::Vector3d delta(x - prev_x, y - prev_y, 0.0);

	Eigen::Vector3d frontVec = (lookAt - eye).normalized();
	Eigen::Vector3d upVec = up.normalized();
	Eigen::Vector3d rightVec = frontVec.cross(upVec).normalized();

	delta = (-rightVec * delta.x() + upVec * delta.y())* (lookAt - eye).norm()/1200.0;

	lookAt += delta; eye += delta;
}
