#include "Environment.h"
#include <iostream>
#include <chrono>

using namespace std;

Environment::
Environment(int control_Hz, int simulation_Hz, int numChars)
:mControlHz(control_Hz), mSimulationHz(simulation_Hz), mNumChars(numChars), mWorld(std::make_shared<dart::simulation::World>())
{

	// Create A team, B team players.
	for(int i=0;i<numChars/2;i++)
	{
		mCharacters[i] = new Character2D("A_" + to_string(i));
		mCharacters[numChars/2+i] = new Character2D("B_" + to_string(i));
	}

	// Add skeletons
	for(int i=0;i<numChars;i++)
	{
		mWorld->addSkeleton(mCharacters[i]->getSkeleton());
	}
}

void
Environment::
step()
{
	mWorld->step();
}