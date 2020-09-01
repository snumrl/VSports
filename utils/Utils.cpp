#include "Utils.h"

Eigen::VectorXd
Utils::
toEigenVec(std::vector<double> vec)
{
	Eigen::VectorXd eigen(vec.size());
	for(int i=0;i<vec.size();i++)
	{
		eigen[i] = vec[i];
	}
	return eigen;
}

std::vector<double>
Utils::
toStdVec(Eigen::VectorXd eigen)
{
	std::vector<double> vec;
	vec.resize(eigen.rows());
	for(int i=0;i<vec.size();i++)
	{
		vec[i] = eigen[i];
	}
	return vec;
}

