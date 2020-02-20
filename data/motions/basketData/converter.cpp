#include <iostream>
#include <string>
#include <fstream>
using namespace std;

class ICA_ANN
{
public:
	ICA_ANN(){}

	string path;
	int startFrame;
	int endFrame;
	string action;
	int interactionFrame;
};

string getFileName_(const char* path)
{
	string pathString; pathString = path;
	int cur = 0;
	while(pathString.find("/") != std::string::npos)
	{
		pathString.erase(0, pathString.find("/")+1);
	}

	return pathString.substr(0, pathString.find("."));
}


int main(int argc, char** argv)
{
	string filePath;
	if(argc == 2)
		filePath = argv[1];
	// string writeFilePath = "./ann/"+getFileName_(filePath.data())+".ann";
	string writeFilePath = "./ann/0_ann";

	ifstream openFile(filePath.data());
	ofstream writeFile(writeFilePath.data(), ios::app);

	if( !openFile.is_open() )
	{
		cout<<"Failed open file "<<filePath<<endl;
		return 0;
	}
	if( !writeFile.is_open() )
	{
		cout<<"Failed write file "<<writeFilePath<<endl;
		return 0;
	}

	string line;
	// {
	// defaultType:b_right
	// file:s_003_1_1.bvh
	// person:1
	// oppositePerson:0
	// startFrame:170
	// endFrame:189
	// include:
	// isAlone:
	// type:dribble
	// subtype:
	// mood:
	// power:
	// beforeActiveState:
	// afterActiveState:
	// beforePassiveState:
	// afterPassiveState:
	// mutipleInteraction:
	// interactionFrame:-1
	// interactionActivePart:
	// interactionPassivePart:
	// interactionType:
	// weight:1.0
	// },
	while(getline(openFile, line))	// {

	{
		ICA_ANN annFormat;
		if(line[0] == '{')
		{

			getline(openFile, line);	// defaultType:b_right

			getline(openFile, line);	// file:s_003_1_1.bvh
			line.erase(0,line.find(":")+1);
			annFormat.path = line.substr(0,line.size()-1);

			getline(openFile, line);	// person:1

			getline(openFile, line);	// oppositePerson:0

			getline(openFile, line);	// startFrame:170
			line.erase(0,line.find(":")+1);
			annFormat.startFrame = atoi(line.data());

			getline(openFile, line);	// endFrame:189
			line.erase(0,line.find(":")+1);
			annFormat.endFrame = atoi(line.data());

			getline(openFile, line);	// include:

			getline(openFile, line);	// isAlone:

			getline(openFile, line);	// type:dribble
			line.erase(0,line.find(":")+1);
			annFormat.action = line.substr(0,line.size()-1);

			getline(openFile, line);	// subtype:

			getline(openFile, line);	// mood:

			getline(openFile, line);	// power:

			getline(openFile, line);	// beforeActiveState:

			getline(openFile, line);	// afterActiveState:

			getline(openFile, line);	// beforePassiveState:

			getline(openFile, line);	// afterPassiveState:

			getline(openFile, line);	// mutipleInteraction:

			getline(openFile, line);	// interactionFrame:-1
			line.erase(0,line.find(":")+1);
			annFormat.interactionFrame = atoi(line.data());

			getline(openFile, line);	// interactionActivePart:

			getline(openFile, line);	// interactionPassivePart:

			getline(openFile, line);	// interactionType:

			getline(openFile, line);	// weight:1.0

			getline(openFile, line);	// },

			writeFile<<annFormat.path<<" "<<annFormat.startFrame<<" "<<annFormat.endFrame<<" ";
			writeFile<<annFormat.action<<" "<<annFormat.interactionFrame<<endl;

		}
		else
		{
			cout<<"Please check the file format"<<endl;
		}

	}
	
	openFile.close();
	writeFile.close();

}