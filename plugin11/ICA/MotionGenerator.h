#ifndef __EXTERN_ICA_MOTION_GENERATOR_H__
#define __EXTERN_ICA_MOTION_GENERATOR_H__

#include <GL/glew.h>
#include <GL/glu.h>
#include <GL/freeglut.h>

#include "ICA/Motion/MotionSegment.h"
#include "ICA/rnn2/RunTimeMotionGenerator.h"

namespace ICA_MOTIONGEN
{
    class MotionGenerator
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        MotionGenerator(std::string dir, std::string guide_x_dir="");
        ~MotionGenerator();

        int frame;
        bool isPlaying;

        std::vector<RunTimeMotionGenerator*> motionGenerators;

        bool setGlobalGoal;
        Eigen::Vector2d targetLocal;
        Eigen::Vector2d goal;
        // int type;
        // int nextType;
        // int typeChangeTimeLeft;
        // int criticalTimeLeft;

        bool usePrediction=true;

        std::vector<Eigen::Vector2d> guideXData;
        std::vector<std::vector<double>> guided_x_temp;

        Eigen::VectorXd generateNextPose();
        std::vector<double> prediction;

        std::vector<std::vector<double>> startTargets;

        // void drawGuidePath();

        Eigen::VectorXd mean;
        Eigen::VectorXd std;

        Eigen::Vector3d initialRelativeCoord;
        
        int mTotalTypeNum;
        // void readTime(int ann_type_num, std::string dir, bool start);
        std::map<std::pair<int,int>, std::vector<int>> mStartTimeMap;
        std::map<std::pair<int,int>, std::vector<int>> mEndTimeMap;
        // int critical_point_offset;

        int focusMotion= 0;
    };
}



#endif //ICA_TESTVIEWERWINDOW_H
