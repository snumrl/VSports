#ifndef __GL_FUNCTIONS_H__
#define __GL_FUNCTIONS_H__
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Geometry>
#include <assimp/cimport.h>
#include <assimp/scene.h>
#include "Camera.h"
#include <memory>

namespace GUI
{
	void drawSphere(double r);
	void drawSphere(double r, Eigen::Vector3d position, Eigen::Vector3d color, int slices = 16, int stacks = 16);

	void drawCube(const Eigen::Vector3d& size);
	void drawTetrahedron(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& p2,const Eigen::Vector3d& p3,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8));
	void drawTriangle(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& p2,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8));
	void drawLine(const Eigen::Vector3d& p0,const Eigen::Vector3d& p1,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8));
	void drawPoint(const Eigen::Vector3d& p0,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8));
	void drawCylinder(double _radius, double _height,const Eigen::Vector3d& color = Eigen::Vector3d(0.8,0.8,0.8), int slices = 16, int stacks = 16);
	void drawStringOnScreen(float _x, float _y, const std::string& _s,bool _bigFont,const Eigen::Vector3d& color=Eigen::Vector3d(0.8,0.8,0.8));

	void drawStringOnScreen_small(float _x, float _y, const std::string& _s,const Eigen::Vector3d& color=Eigen::Vector3d(0.8,0.8,0.8));

	void drawMapOnScreen(Eigen::VectorXd minimap, int numRows, int numCols);

	void drawValueGradientBox(Eigen::VectorXd states, Eigen::VectorXd valueGradient, double boxSize = 0.2);
	void drawValueBox(Eigen::VectorXd value, double boxSize = 0.2);

	void drawSquare(double width, double height);
	void drawSoccerLine(double x, double y);
	void drawMesh(const Eigen::Vector3d& scale, const aiScene* mesh,const Eigen::Vector3d& color=Eigen::Vector3d(0.8,0.8,0.8));
	void drawVerticalLine(const Eigen::Vector2d& point, const Eigen::Vector3d& color=Eigen::Vector3d(1.0, 1.0, 1.0));
	void drawCoordinate(Eigen::Vector3d center, double length = 1.0);
	void draw2dCircle(Eigen::Vector3d center, Eigen::Vector3d xAxis, Eigen::Vector3d yAxis, double radius, bool filled);

	void drawAuxLine(double gridSize=1.0, const Eigen::Vector3d& color=Eigen::Vector3d(1.0, 1.0, 1.0));
	// void drawCylinder()
};

Eigen::Vector3d degreeToRgb(double degree, bool forValue = false);

#endif