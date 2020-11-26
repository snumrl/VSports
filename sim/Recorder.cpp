#include "Recorder.h"

Recorder::Recorder()
{

}

void
Recorder::recordCurrentFrame(Environment* env)
{
	skelPositions.push_back(env->mCharacters[0]->getSkeleton()->getPositions());
	obstaclePositions.push_back(env->mObstacles[0]);
	ballPositions.push_back(env->ballSkel->getPositions());
}


void
Recorder::loadFrame(Environment* env, int frame)
{
	if(frame > skelPositions.size()-1)
		frame = skelPositions.size()-1;
	else if(frame < 0)
		frame = 0;
	env->mCharacters[0]->getSkeleton()->setPositions(skelPositions[frame]);
	env->mObstacles[0] = obstaclePositions[frame];
	env->ballSkel->setPositions(ballPositions[frame]);
}

int
Recorder::loadLastFrame(Environment* env)
{
	if(skelPositions.size() == 0)
		return -1;
	loadFrame(env, skelPositions.size()-1);
	return skelPositions.size()-1;
}