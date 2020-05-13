#ifndef __UTILS_H__
#define __UTILS_H__
#include <Eigen/Dense>
#include <iostream>
#include <vector>
namespace Utils
{
Eigen::VectorXd toEigenVec(std::vector<double> vec);
std::vector<double> toStdVec(Eigen::VectorXd eigen);
}

#endif