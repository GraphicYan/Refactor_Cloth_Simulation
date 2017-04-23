#pragma once
#include "LoadObj.h"

class Collision_obj
{
public:
	Collision_obj(Obj& obj);
public:
	vector<glm::vec3> vertices;
	vector<int> vertexIndices;

	vector<glm::vec3> vertices_in_faceseq;
	vector<glm::vec3> barys;
public:
	void scale(float s);
};
