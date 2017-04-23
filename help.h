#pragma once
#include <stdlib.h>

void getFPS();

void GetPath(char retpath[], const char *file, char *file_ext); //get the absolute path of .exe

void PickUpName(char name[_MAX_DIR], char OldPath[_MAX_DIR]);