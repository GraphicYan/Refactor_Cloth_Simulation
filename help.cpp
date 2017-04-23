#include "stdafx.h"
#include <GL/freeglut.h>
#include "help.h"
#include <time.h>
#include <string>
using namespace std;
 
float startTime = 0;
int totalFrames = 0;
float currentTime = 0;
char info[MAX_PATH] = { 0 };

void getFPS()
{
	float fps = 0;
	float newTime = (float)glutGet(GLUT_ELAPSED_TIME);
	float frameTime = newTime - currentTime;
	currentTime = newTime;
	++totalFrames;
	if ((newTime - startTime)>1000)
	{
		float elapsedTime = (newTime - startTime);
		fps = (totalFrames / elapsedTime) * 1000;
		startTime = newTime;
		totalFrames = 0;

		sprintf_s(info, "GLUT Cloth Demo FPS: %4.3f", fps);
	}
	glutSetWindowTitle(info);
}


void GetPath(char retpath[], const char *file, char *file_ext)
{

	char drive[_MAX_DRIVE];
	char dir[_MAX_DIR];
	char fname[_MAX_FNAME];
	char ext[_MAX_EXT];

	_splitpath(_pgmptr, drive, dir, fname, ext); // C4996
												 //strcat(dir,"Data\\");
	_makepath(retpath, drive, dir, file, file_ext); // C4996
}

void PickUpName(char name[_MAX_DIR], char OldPath[_MAX_DIR])
{
	string path(OldPath);

	path = path.substr(0, path.size() - 4);

	int i;
	for (i = 0; i<path.length(); i++)
		name[i] = path[i];
	name[i] = '\0';
}