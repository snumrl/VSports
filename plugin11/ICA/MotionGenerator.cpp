#include "MotionGenerator.h"
// #include "Painter.h"
#include <iostream>
#include "ICA/CharacterControl/MotionRepresentation.h"
#include "ICA/Utils/PathManager.h"
#include "ICA/Utils/Functions.h"
#include "ICA/Motion/RootTrajectory.h"

#include "ICA/CharacterControl/RunJumpRollControlObjective.h"
#include "ICA/CharacterControl/RunJumpRollControlObjective_wtiming.h"
#include "ICA/CharacterControl/Dir_Action_Time_ControlObjective.h"
#include "ICA/CharacterControl/TimeControlObjective.h"
#include "ICA/Motion/MotionSegment.h"
#include "ICA/rnn2/RunTimeMotionGenerator.h"



//#include "RunTimeMotionGenerator.h"
using namespace ICA_MOTIONGEN;
namespace p = boost::python;
namespace np = boost::python::numpy;
using namespace std;
#define DEFAULT_ACTION 7
#define TOTAL_ACTION_NUM 8

// void TestViewerWindow::readTime(int ann_type_num, std::string dir, bool start)
// {
//     std::string fileName= (start)? "startTimeCheck" : "endTimeCheck";
//     auto& map = (start)? mStartTimeMap : mEndTimeMap;

//     std::string timePath= PathManager::getFilePath_data(dir, "data", fileName);
//     std::ifstream timeCheckFile(timePath, std::ios::out );
//     std::cout<<timePath<<std::endl;
    
//     // for(int startType= 0; startType< ann_type_num; startType++)
//     // {
//     //     for(int endType= 0; endType< ann_type_num; endType++)
//     //     {
//     //         if(startType == endType) continue;
            
//     std::string buffer;

//     timeCheckFile>> buffer;
//     int startType= atoi(buffer.c_str());
//     // Basic::myAssert(buffer == std::to_string(startType), "buffer ("+buffer+") == startType("+std::to_string(startType)+")", __FILE__, __LINE__, __func__);
//     timeCheckFile>> buffer;
//     int endType= atoi(buffer.c_str());
//     // Basic::myAssert(buffer == std::to_string(endType), "buffer ("+buffer+") == startType("+std::to_string(endType)+")", __FILE__, __LINE__, __func__);

//     timeCheckFile>> buffer;
//     int timeNum= atoi(buffer.c_str());
//     std::cout<<startType<<" "<<endType<<" : num "<<timeNum<<std::endl;
//     std::vector<int> time;
//     for(int i=0; i<timeNum; i++)
//     {
//         timeCheckFile>>buffer;
//         time.push_back(atoi(buffer.c_str()));
//         std::cout<<buffer<<" ";
//     }
//     if(timeNum!=0) std::cout<<std::endl;

//     map[std::make_pair(startType, endType)]= time;
// //     }
//     // }   
// }


MotionGenerator::MotionGenerator(std::string dir, std::string guide_x_dir)
{
    // this->mCamera->mTrackCamera= true; // false;


    this->mTotalTypeNum= TOTAL_ACTION_NUM;
    // this->readTime(mTotalTypeNum, dir,true);
    // this->readTime(mTotalTypeNum, dir,false);

    int startSize= 50;
    std::vector<double> startData;

    std::string yNormalPath=  PathManager::getFilePath_training_data(dir, "yNormal.dat");
    std::string yDataPath=  PathManager::getFilePath_training_data(dir, "yData.dat");
    cout<<yNormalPath<<" "<<yDataPath<<endl;
    MotionRepresentation::getStartingDataAndTargets(dir+"/"+"start_walk", startSize, startData, startTargets);
    MotionRepresentation::readXNormal(dir+"/"+"start_walk", mean, std);

    std::cout<<mean.transpose()<<std::endl;
    std::cout<<"mean target: "<<mean[0]<<", "<<mean[1]<<std::endl;
    std::cout<<"std target: "<<std[0]<<", "<<std[1]<<std::endl;

    std::cout<<"start? "<<std::endl;
    Py_Initialize();
    np::initialize();
    PyRun_SimpleString("import sys");
    PyRun_SimpleString("sys.path.append(\".\")");
    // PyRun_SimpleString("import tensorflow as tf");
    // PyRun_SimpleString("with tf.device('/cpu:0'):");


//    this->rtmg->co = new WalkControlObjective();
    this->usePrediction= false; //true;

    std::vector<double> next_y;
    bool loadDone = false;
    while(! loadDone)
    {
        try
        {
            std::string input, input2;

            std::cout<<"type directory or \'done\'"<<std::endl;
            std::cin>>input;
            
            if(input!= "done") 
            {
                std::cout<<"type network_index or \'done\'"<<std::endl;
                std::cin>>input2;

                int network_index= std::stoi(input2);

                RunTimeMotionGenerator* rtmg= new RunTimeMotionGenerator(input, network_index);
                rtmg->setStartPose(startData);

                std::cout<<"start:? "<<std::endl;

                for(int i=0; i<startSize-1; i++)
                {
                    std::vector<double> target= startTargets[i]; //(i==0)? Basic::toStdVec(mean): prediction;// {mean[0], mean[1]};
                    // std::cout<<"initializing/ startTargets: ";
                    // for(int i=0; i<target.size(); i++) std::cout<<target[i]<<" ";
                    // std::cout<<std::endl;
                    if(usePrediction) rtmg->getNextPose(target, next_y, prediction);
                    else rtmg->getNextPose(target, next_y);

                    for(int i=0; i<next_y.size(); i++)
                    {
                        std::cout<<next_y[i]<<" ";
                        if(i==5 || i==(5+48-1)) std::cout<<std::endl;
                    }
                    std::cout<<std::endl;
                }
                motionGenerators.push_back(rtmg);
            }
            else loadDone= true;
        }
        catch(p::error_already_set const &)
        {
            std::cout<<"error here "<<__FILE__<<"/ "<<__LINE__<<"/"<<std::endl;
            PyErr_Print();// handle the exception in some way
        }
    }


    std::cout<<"after setting startSize"<<std::endl;

    std::vector<double> target= startTargets[startSize-1]; //Basic::toStdVec(mean);

    for(auto rtmg: this->motionGenerators)
    {
        rtmg->mMotionSegment->clearPose(); ;
        Motion::Pose* startPose= rtmg->getNextPose(target, next_y);        
        // rtmg->mMotionSegment->addPose(startPose);
    }
    std::cout<<__func__<<" / chkpt 1"<<std::endl;

    this->frame=0;
    this->isPlaying= false;
    this->goal= Eigen::Vector2d(0,0);

    if(guide_x_dir!="")
    {
        std::string guide_dir= PathManager::getDirPath_training_data(guide_x_dir);
        this->guideXData= MotionRepresentation::getXGlobalData(guide_dir);
    }

    std::cout<<__func__<<" / done"<<std::endl;
}


Eigen::VectorXd MotionGenerator::generateNextPose()
{
    int cnt=0;
    for(auto rtmg: this->motionGenerators)
    {
        auto mMotionSegment= rtmg->mMotionSegment;
        if(frame+1 < mMotionSegment->mPoses.size()) frame= mMotionSegment->mPoses.size()-1;
        std::cout<<"frame: "<<frame<<std::endl;
        Motion::Pose *endPose = mMotionSegment->getLastPose(); //mPoses[this->mMotionSegment->mPoses.size()-1];
        Eigen::Vector2d basePos = endPose->getRoot().pos;
        Eigen::Vector2d baseDir = endPose->getRoot().dir;

        std::cout<<"goal ; "<<goal.transpose()<<std::endl;
    //    Eigen::Vector2d targetLocal;
        if(setGlobalGoal)
        {
            targetLocal = Motion::Root::getRelativeCoord(baseDir, basePos, goal);
            // targetLocal.normalize();
            // std::cout<<"targetLocal: "<<targetLocal.transpose()<<std::endl;
        }
        else
        {
            // restore global goal
            Eigen::Vector2d baseDir_R= Eigen::Vector2d(-baseDir[1], baseDir[0]);
            goal = basePos + baseDir* targetLocal[0] + baseDir_R* targetLocal[1];
        }

        std::cout<<"targetLocal ; "<<targetLocal.transpose()<<std::endl;
        
        try
        {
            std::vector<double> targetLocal_vec= {targetLocal[0], targetLocal[1]};

            prediction= std::vector<double>();        
            std::vector<double> next_y= std::vector<double>();
        
            std::vector<double> prevPoseData;
            MotionRepresentation::getData(mMotionSegment, prevPoseData, mMotionSegment->mPoses.size()-1);

            Motion::Pose *newPose = rtmg->getNextPose(targetLocal_vec, prevPoseData, next_y, false);
            for(auto& jointAngle :newPose->mJointAngles)
            {
                cout<<jointAngle.linear()<<endl;
            }
            cout<<endl;
            
            cnt++;
        }
        catch(p::error_already_set const &)
        {
            PyErr_Print();// handle the exception in some way
        }
    }

    this->frame++;
    // this->graphWindow->currentFrame++;

    return Eigen::Vector3d::Zero();
    
}

MotionGenerator::~MotionGenerator()
{
}

/*
void TestViewerWindow::display()
{
//    std::cout<<"TestViewerWindow/display: "<<glutGetWindow()<<std::endl;
    glClearColor(0.5, 0.5, 0.5, 1);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);

    glPushMatrix();
    {
        glTranslatef(this->goal[0], 0, this->goal[1]);
        glRotatef(-90, 1,0,0);
        glColor3f(0.8,0.8,1);
        drawBox(500, 1);
    }
    glPopMatrix();

    Eigen::Vector3d scaled_center(0,0,0);

    if(this->mCamera->mTrackCamera)
    {
        Eigen::Vector2d skeleton_root= motionGenerators[focusMotion]->mMotionSegment->getPose(frame)->getRoot().pos;
        scaled_center= Eigen::Vector3d(0.1*skeleton_root[0], 0, 0.1*skeleton_root[1]);
        this->mCamera->SetCenter(scaled_center);
//        initLights(scaled_center[0], scaled_center[2]);
    }
    DrawGround(scaled_center, 500, 10000,10000,1);
    drawCoordinate(scaled_center);

    mCamera->Apply();

//    drawCoordinate();
    glScaled(0.1, 0.1, 0.1);

    int cnt= 0;
    for(auto rtmg: motionGenerators)
    {
        // std::cout<<"rtmg- pose: "<<rtmg->mMotionSegment->mPoses.size()<<", frame= "<<frame<<std::endl;
        // Eigen::Vector3d color= GUI::getAnnColor(cnt, motionGenerators.size());
        // glColor3f(color[0], color[1], color[3]);

        drawPose(rtmg->mMotionSegment->getPose(frame));
    
        Eigen::Vector2d basePos= (frame>0)? rtmg->mMotionSegment->getPose(frame-1)->getRoot().pos: Eigen::Vector2d(0,0);
        Eigen::Vector2d baseDir= (frame>0)? rtmg->mMotionSegment->getPose(frame-1)->getRoot().dir: Eigen::Vector2d(1,0);

        glPushMatrix();
        glBegin(GL_LINES);
        glVertex3f(basePos[0], 0, basePos[1]);
        glVertex3f((basePos+300*baseDir)[0], 0, (basePos+300*baseDir)[1]);
        glEnd();
        
        cnt++;
    } 

    glPopMatrix();


    glutSwapBuffers();


}
*/