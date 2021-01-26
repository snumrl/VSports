#include "Recorder.h"

Recorder::Recorder()
{
	skelPositions.resize(1);
	obstaclePositions.resize(0);
}

void
Recorder::recordCurrentFrame(Environment* env)
{
	for(int i=0;i<skelPositions.size();i++)
	{
		skelPositions[i].push_back(env->mCharacters[i]->getSkeleton()->getPositions());
	}
	for(int i=0;i<obstaclePositions.size();i++)
	{
		obstaclePositions[i].push_back(env->mObstacles[i]);
	}
	ballPositions.push_back(env->ballSkel->getPositions());
}


void
Recorder::loadFrame(Environment* env, int frame)
{
	if(frame > skelPositions[0].size()-1)
		frame = skelPositions[0].size()-1;
	else if(frame < 0)
		frame = 0;

	for(int i=0;i<skelPositions.size();i++)
	{
		env->mCharacters[i]->getSkeleton()->setPositions(skelPositions[i][frame]);
	}
	for(int i=0;i<obstaclePositions.size();i++)
	{
		env->mObstacles[i] = obstaclePositions[i][frame];
	}
	env->ballSkel->setPositions(ballPositions[frame]);
}

int
Recorder::loadLastFrame(Environment* env)
{
	if(skelPositions[0].size() == 0)
		return -1;
	loadFrame(env, skelPositions[0].size()-1);
	return skelPositions.size()-1;
}