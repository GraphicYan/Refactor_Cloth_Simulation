#include "stdafx.h"
#include "collision_body.h"

Collision_obj::Collision_obj(Obj& obj)
{
	vertices = obj.vertices;
	vertexIndices = obj.vertexIndices;

	for (int i = 0; i < vertexIndices.size(); i+=3)
	{
		glm::vec3 pos[3];
		for (int k=0;k<3;k++)
		{
			pos[k] = vertices[vertexIndices[i+k]];
			vertices_in_faceseq.push_back(pos[k]);          //按照面的索引放置点
		}
		glm::vec3 centric = (pos[0] + pos[1] + pos[2]);
		centric /= 3.0;
		barys.push_back(centric);
	}
}

void Collision_obj::scale(float S)
{
	int n = vertices.size();
	float sumx = 0, sumy = 0, sumz = 0;
	float avex, avey, avez;
	for (int i = 0; i<n; ++i)
	{
		sumx += vertices[i].x;
		sumy += vertices[i].y;
		sumz += vertices[i].z;
	}
	avex = sumx / n;
	avey = sumy / n;
	avez = sumz / n;

	//const float up = 1.2;
	for (int i = 0; i<n; ++i)
	{
		//glm::vec3 vertex = glm::vec3(data.VertexArray[i].X-avex,data.VertexArray[i].Y-avey,data.VertexArray[i].Z-avez);
		vertices[i].x = (vertices[i].x - avex) * S; vertices[i].x += avex;
		vertices[i].y = (vertices[i].y - avey) * S; vertices[i].y += avey;
		vertices[i].z = (vertices[i].z - avez) * S; vertices[i].z += avez;
	}
}