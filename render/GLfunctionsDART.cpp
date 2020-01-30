#include "GLfunctionsDART.h"
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace std;
void
GUI::
drawSkeleton(
		const dart::dynamics::SkeletonPtr& skel,
		const Eigen::Vector3d& color,
		bool mesh,
		bool wireFrame)
{
	for(int i=0;i<skel->getNumBodyNodes();i++)
	{
		auto bn = skel->getBodyNode(i);
		auto shapeNodes = bn->getShapeNodesWith<VisualAspect>();
		int j = (mesh? 1:0);
		if(shapeNodes.size() ==1)
			j=0;
		auto T = shapeNodes[j]->getTransform();
		drawShape(T,shapeNodes[j]->getShape().get(),color, wireFrame);
	}

}


void
GUI::
drawShape(const Eigen::Isometry3d& T,
	const dart::dynamics::Shape* shape,
	const Eigen::Vector3d& color,
	bool wireFrame)
{
	glEnable(GL_LIGHTING);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glColor3f(color[0],color[1],color[2]);
	glPushMatrix();
	glMultMatrixd(T.data());
	if(shape->is<SphereShape>())
	{
		const auto* sphere = dynamic_cast<const SphereShape*>(shape);
		if(wireFrame)
			glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
		else
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);		
		GUI::drawSphere(sphere->getRadius());
		// glColor3f(0,0,0);
		// glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
		// GUI::DrawSphere(sphere->getRadiusLength());
	}
	else if (shape->is<BoxShape>())
	{
		const auto* box = dynamic_cast<const BoxShape*>(shape);
		if(wireFrame)
			glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
		else
			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);	
		GUI::drawCube(box->getSize());
    	// GUI::DrawCube(Eigen::Vector3d(0.01,0.01,0.01));
	}
	else if(shape->is<MeshShape>())
	{
		auto* mesh = dynamic_cast<const MeshShape*>(shape);
		// for(int i =0;i<16;i++)
			// std::cout<<(*mesh->getMesh()->mRootNode->mTransformation)[i]<<" ";
		glDisable(GL_COLOR_MATERIAL); 
    	GUI::drawMesh(mesh->getScale(),mesh->getMesh());

	}

	glPopMatrix();
}




// void
// GUI::
// DrawShape(const Eigen::Isometry3d& T,
// 	const dart::dynamics::Shape* shape,
// 	const Eigen::Vector3d& color)
// {
// 	glEnable(GL_LIGHTING);
// 	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
// 	glEnable(GL_COLOR_MATERIAL);
// 	glColor3f(color[0],color[1],color[2]);
// 	glPushMatrix();
// 	glMultMatrixd(T.data());
// 	if(shape->is<SphereShape>())
// 	{
// 		const auto* sphere = dynamic_cast<const SphereShape*>(shape);
// 		if(wireFrame)
// 			glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
// 		else
// 			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);		
// 		GUI::drawSphere(sphere->getRadius());
// 		// glColor3f(0,0,0);
// 		// glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
// 		// GUI::DrawSphere(sphere->getRadiusLength());
// 	}
// 	else if (shape->is<BoxShape>())
// 	{
// 		const auto* box = dynamic_cast<const BoxShape*>(shape);
// 		if(wireFrame)
// 			glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
// 		else
// 			glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);	
// 		GUI::drawCube(box->getSize());
//     	// GUI::DrawCube(Eigen::Vector3d(0.01,0.01,0.01));
// 	}
// 	else if(shape->is<MeshShape>())
// 	{
// 		auto* mesh = dynamic_cast<const MeshShape*>(shape);

// 		// for(int i =0;i<16;i++)
// 			// std::cout<<(*mesh->getMesh()->mRootNode->mTransformation)[i]<<" ";
//     	GUI::drawMesh(mesh->getScale(),mesh->getMesh());

// 	}

// 	glPopMatrix();

// 	// glDisable(GL_COLOR_MATERIAL);
// }