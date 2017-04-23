#include "stdafx.h"
#include "LoadObj.h"
#include <vector>
#include <iostream>
#include <GL/glaux.H>

using namespace std;

#pragma comment(lib, "glaux.lib")

bool read_mtl(char *mtlpath, vector<Material> &all_mtl)
{
	FILE * file_mtl = fopen(mtlpath, "r");
	if (file_mtl == NULL)
	{
		printf("Impossible to open the .mtl file ! Plese check the path .\n");
	}

	while (1)
	{
		char lineHeader[128];  //read the first word of the line
		int res = fscanf(file_mtl, "%s", lineHeader);
		if (res == EOF)
		{
			fclose(file_mtl);
			return false;
		}

		if (strcmp(lineHeader, "newmtl") == 0) break;
		else
		{
			// Probably a comment, eat up the rest of the line
			char stupidBuffer[1000];
			fgets(stupidBuffer, 1000, file_mtl);
		}
	}

	while (1)
	{
		Material new_mtl;
		fscanf(file_mtl, "%s", &new_mtl.mtl_name);
		int status = 1;

		while (1)
		{
			char parameter[32];
			status = fscanf(file_mtl, "%s", &parameter);
			if (status == EOF)
			{
				fclose(file_mtl);
				break;
			}

			if (strcmp(parameter, "Ns") == 0)
			{
				fscanf(file_mtl, "%f\n", &new_mtl.Ns);
			}
			else if (strcmp(parameter, "Ni") == 0)
			{
				fscanf(file_mtl, "%f\n", &new_mtl.Ni);
			}
			else if (strcmp(parameter, "d") == 0)
			{
				fscanf(file_mtl, "%f\n", &new_mtl.d);
			}
			else if (strcmp(parameter, "Tr") == 0)
			{
				fscanf(file_mtl, "%f\n", &new_mtl.Tr);
			}
			else if (strcmp(parameter, "Tf") == 0)
			{
				fscanf(file_mtl, "%f\n", &new_mtl.Tf);
			}
			else if (strcmp(parameter, "illum") == 0)
			{
				fscanf(file_mtl, "%d\n", &new_mtl.illum);
			}
			else if (strcmp(parameter, "Ka") == 0)
			{

				fscanf(file_mtl, "%f %f %f\n", &new_mtl.Ka.x, &new_mtl.Ka.y, &new_mtl.Ka.z);

			}
			else if (strcmp(parameter, "Kd") == 0)
			{

				fscanf(file_mtl, "%f %f %f\n", &new_mtl.Kd.x, &new_mtl.Kd.y, &new_mtl.Kd.z);

			}
			else if (strcmp(parameter, "Ks") == 0)
			{

				fscanf(file_mtl, "%f %f %f\n", &new_mtl.Ks.x, &new_mtl.Ks.y, &new_mtl.Ks.z);

			}
			else if (strcmp(parameter, "Ke") == 0)
			{

				fscanf(file_mtl, "%f %f %f\n", &new_mtl.Ke.x, &new_mtl.Ke.y, &new_mtl.Ke.z);

			}
			else if ((strcmp(parameter, "map_Kd") == 0) || (strcmp(parameter, "map_Ka") == 0))
			{

				char path[_MAX_DIR], name[_MAX_DIR];
				fscanf(file_mtl, "%s", path);
				PickUpName(name, path);
				GetPath(new_mtl.text_path, name, "bmp");

			}
			else if (strcmp(parameter, "newmtl") == 0)
			{
				//all_mtl.push_back(new_mtl);
				break;
			}
			else
			{
				// Probably a comment, eat up the rest of the line
				char stupidBuffer[1000];
				fgets(stupidBuffer, 1000, file_mtl);
			}
		}
		all_mtl.push_back(new_mtl);
		if (status == EOF) break;
	}


	for (int i = 0; i<all_mtl.size(); i++)               //绑定贴图
	{
		bind_mtl(all_mtl[i].text_path,all_mtl[i].clothTexture);
	}
	return true;
}


bool find_mtl(char *mtl_name, vector<Material> all_mtl, Material &mtl)
{
	for (int i = 0; i<all_mtl.size(); i++)
	{
		if (strcmp(all_mtl[i].mtl_name, mtl_name) == 0)
		{
			mtl = all_mtl[i];
			return true;
		}
	}
	return false;
}


bool bind_mtl(char *mtl_path,GLuint& clothTexture)
{
	AUX_RGBImageRec *pImage;
	pImage = auxDIBImageLoadA(mtl_path);

	glGenTextures(1,&clothTexture);  
	glBindTexture(GL_TEXTURE_2D,clothTexture);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);//此为纹理过滤参数设置
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, 3, pImage->sizeX, pImage->sizeY, 0, GL_RGB, GL_UNSIGNED_BYTE,pImage->data);
	glGenerateMipmap(GL_TEXTURE_2D);//启用MIP贴图
	glBindTexture(GL_TEXTURE_2D,0);

	return true;
}


Group::Group(int f, int l, Material mtl)
{
	IF_ELASTICITY = 1;   //正常情况  初始化为有弹性部分
	First = f;
	Last = l;
	Group_Material = mtl;
	Group_size = l - f + 1;
}

bool Group::set_tri(int first_tri, int last_tri)
{
	First_Tri = first_tri;
	Last_Tri = last_tri;
	return true;
}



Obj::Obj(ObjType objType, const char * clothpath)
{
	load(objType,clothpath);
}




bool Obj::RemoveBack(){
	for(int i=0;i<groups.size();i+=3)
	{
		cout<<groups[i+1].First<<"  :   "<<groups[i+1].Last<<endl;
		for(int n=groups[i+1].First;n<=groups[i+1].Last;n++)
			vertices[n]=glm::vec3(-100.0f);	
	}
	return 1;
}

bool Obj::load(ObjType objType, const char * obj_path)
{
	char abs_obj_path[_MAX_DIR];
	GetPath(abs_obj_path, obj_path, "obj");

	if (objType == CLOTH)
	{	loadOBJ_cloth(abs_obj_path);
		change_size(0.31, 0, 1.935, -0.23);
		RemoveBack(); //去除内层影响
	}
	else if (objType == BODY)
	{
		loadOBJ_body(abs_obj_path);
		change_size(0.32, 0, 2.25, -0.1);    //collision body size
		
	}
	else
	{
		cout << "obj type is not correct!" << endl;
		return false;
	}

	obj_type = objType;


	count_tri();  //计算每个组内首、尾三角面的索引
	uv_normalize();
	normal_normalize(); //归一化坐标、法向量、纹理索引
	set_tri_index(); //计算每个点所在的所有面（每个点固定20个面容量）
	compute_normal(); //重新计算第一次的法向量
	vec3_to_vec4(vertices, vertices4);   // for shader

	return true;
}

bool Obj::loadOBJ_cloth(const char * clothpath)
{
	printf("Loading Cloth file %s for simulation.\n", clothpath);
	FILE * file_obj = fopen(clothpath, "r");

	if (file_obj == NULL)
	{
		printf("Impossible to open the .obj file ! Plese check the path .\n");
		getchar();
		return false;
	}

	vector<Material> all_mtl;
	int first = 0, last = 0;

	while (1)
	{
		char lineHeader[128];  //read the first word of the line
		int res = fscanf(file_obj, "%s", lineHeader);
		if (res == EOF) break;   //End Of File: Quit the loop.

		if (strcmp(lineHeader, "v") == 0)
		{
			glm::vec3 vertex;
			fscanf(file_obj, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			vertices.push_back(vertex);
		}
		else if (strcmp(lineHeader, "vt") == 0)
		{
			glm::vec2 uv;
			fscanf(file_obj, "%f %f\n", &uv.x, &uv.y);
			uv.y = uv.y; // Invert V coordinate since we will only use DDS texture, which are inverted. Remove if you want to use TGA or BMP loaders.
			uvs.push_back(uv);
		}
		else if (strcmp(lineHeader, "vn") == 0)
		{
			glm::vec3 normal;
			fscanf(file_obj, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
			normals.push_back(normal);
		}
		else if (strcmp(lineHeader, "f") == 0)
		{
			std::string vertex1, vertex2, vertex3;
			unsigned int vI[3], uI[3], nI[3];
			int matches = fscanf(file_obj, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vI[0], &uI[0], &nI[0], &vI[1], &uI[1], &nI[1], &vI[2], &uI[2], &nI[2]);
			if (matches != 9) {
				printf("File can't be read by our simple parser :-( Try exporting with other options\n");
				fclose(file_obj);
				return false;
			}
			vertexIndices.push_back(vI[0] - 1);
			vertexIndices.push_back(vI[1] - 1);
			vertexIndices.push_back(vI[2] - 1);
			uvIndices.push_back(uI[0] - 1);
			uvIndices.push_back(uI[1] - 1);
			uvIndices.push_back(uI[2] - 1);
			normalIndices.push_back(nI[0] - 1);
			normalIndices.push_back(nI[1] - 1);
			normalIndices.push_back(nI[2] - 1);
		}
		else if (strcmp(lineHeader, "usemtl") == 0)
		{
			char mtl_name[128];
			fscanf(file_obj, "%s", &mtl_name);
			Material mtl;
			bool tag = find_mtl(mtl_name, all_mtl, mtl);

			last = vertices.size() - 1;
			Group group(first, last, mtl);
			groups.push_back(group);
			first = vertices.size();
		}
		else if (strcmp(lineHeader, "mtllib") == 0)
		{
			char mtl_path[_MAX_DIR], old_path[_MAX_DIR], name[_MAX_DIR];
			fscanf(file_obj, "%s", &old_path);

			PickUpName(name, old_path);
			GetPath(mtl_path, name, ".mtl");
			read_mtl(mtl_path, all_mtl);
		}
		else {
			// Probably a comment, eat up the rest of the line
			char stupidBuffer[1000];
			fgets(stupidBuffer, 1000, file_obj);
		}

	}

	fclose(file_obj);

	return true;
	
}

bool Obj::loadOBJ_body(const char * bodypath)
{

	printf("Loading Body file %s for simulation.\n", bodypath);
	FILE * file_obj = fopen(bodypath, "r");

	if (file_obj == NULL)
	{
		printf("Impossible to open the .obj file ! Plese check the path .\n");
		getchar();
		return false;
	}

	vector<Material> all_mtl;
	Material mtl;


	while (1)
	{
		char lineHeader[128];  //read the first word of the line
		int res = fscanf(file_obj, "%s", lineHeader);
		if (res == EOF) break;   //End Of File: Quit the loop.

		if (strcmp(lineHeader, "v") == 0)
		{
			glm::vec3 vertex;
			fscanf(file_obj, "%f %f %f\n", &vertex.x, &vertex.y, &vertex.z);
			vertices.push_back(vertex);
		}
		else if (strcmp(lineHeader, "vt") == 0)
		{
			glm::vec2 uv;
			fscanf(file_obj, "%f %f\n", &uv.x, &uv.y);
			uv.y = uv.y; 
			uvs.push_back(uv);
		}
		else if (strcmp(lineHeader, "vn") == 0)
		{
			glm::vec3 normal;
			fscanf(file_obj, "%f %f %f\n", &normal.x, &normal.y, &normal.z);
			normals.push_back(normal);
		}
		else if (strcmp(lineHeader, "f") == 0)
		{
			std::string vertex1, vertex2, vertex3;
			unsigned int vI[3], uI[3], nI[3];
			int matches = fscanf(file_obj, "%d/%d/%d %d/%d/%d %d/%d/%d\n", &vI[0], &uI[0], &nI[0], &vI[1], &uI[1], &nI[1], &vI[2], &uI[2], &nI[2]);
			if (matches != 9) {
				printf("File can't be read by our simple parser :-( Try exporting with other options\n");
				fclose(file_obj);
				return false;
			}
			vertexIndices.push_back(vI[0] - 1);
			vertexIndices.push_back(vI[1] - 1);
			vertexIndices.push_back(vI[2] - 1);
			uvIndices.push_back(uI[0] - 1);
			uvIndices.push_back(uI[1] - 1);
			uvIndices.push_back(uI[2] - 1);
			normalIndices.push_back(nI[0] - 1);
			normalIndices.push_back(nI[1] - 1);
			normalIndices.push_back(nI[2] - 1);
		}
		else if (strcmp(lineHeader, "usemtl") == 0)
		{
			char mtl_name[128];
			fscanf(file_obj, "%s", &mtl_name);
			bool tag = find_mtl(mtl_name, all_mtl, mtl);

		}
		else if (strcmp(lineHeader, "mtllib") == 0)
		{
			char mtl_path[_MAX_DIR], old_path[_MAX_DIR], name[_MAX_DIR];
			fscanf(file_obj, "%s", &old_path);

			PickUpName(name, old_path);
			GetPath(mtl_path, name, ".mtl");
			read_mtl(mtl_path, all_mtl);
		}
		else {
			// Probably a comment, eat up the rest of the line
			char stupidBuffer[1000];
			fgets(stupidBuffer, 1000, file_obj);
		}

	}
	Group group(0, vertices.size() - 1, mtl);
	groups.push_back(group);
	fclose(file_obj);

	return true;
}

bool Obj::change_size(float S, float x_up, float y_up, float z_up)
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
		vertices[i].x = (vertices[i].x - avex) / S; vertices[i].x += x_up;
		vertices[i].y = (vertices[i].y - avey) / S; vertices[i].y += y_up;
		vertices[i].z = (vertices[i].z - avez) / S; vertices[i].z += z_up;
	}
	return true;
}

bool Obj::uv_normalize()
{
	vector<glm::vec2> uvs_old;
	for (int i = 0; i<uvs.size(); i++)
		uvs_old.push_back(uvs[i]);

	uvs.resize(vertices.size());

	for (int i = 0; i<uvIndices.size(); i++)
		uvs[vertexIndices[i]] = uvs_old[uvIndices[i]];
	return true;
}

bool Obj::normal_normalize()
{
	std::vector<glm::vec3> normals_old;
	for (int i = 0; i<normals.size(); i++)
		normals_old.push_back(normals[i]);

	normals.resize(vertices.size());

	for (int i = 0; i<normalIndices.size(); i++)
		normals[vertexIndices[i]] = normals_old[normalIndices[i]];

	return true;
}

bool Obj::count_tri()
{
	int num = 0;
	int first = 0, last = 0;
	for (int i = 0; i<vertexIndices.size(); i += 3)
	{
		if (vertexIndices[i]  > groups[num].getrear())
		{
			last = i - 1;
			groups[num].set_tri(first, last);
			first = i;
			num++;

		}

	}
	groups[num].set_tri(first, vertexIndices.size() - 1);  //
	return true;
}

void Obj::vec3_to_vec4(vector<glm::vec3>& old_vertices, vector<glm::vec4>& new_vertices)
{
	for (auto vec3 : old_vertices)
		new_vertices.emplace_back(vec3.x, vec3.y, vec3.z, 1.0);
}

bool Obj::set_tri_index()
{
	tri_index.resize(vertices.size() * 20);
	for (int i = 0; i<tri_index.size(); i++)  tri_index[i] = -1;
	for (int i = 0; i<vertexIndices.size(); i += 3)
	{
		int pos0 = vertexIndices[i] * 20;
		int pos1 = vertexIndices[i + 1] * 20;
		int pos2 = vertexIndices[i + 2] * 20;
		while (tri_index[pos0] != -1) pos0++;
		while (tri_index[pos1] != -1) pos1++;
		while (tri_index[pos2] != -1) pos2++;
		if (pos0>vertexIndices[i] * 20 + 19) {
			pos0 = vertexIndices[i] * 20 + 19;
			cout << "第" << vertexIndices[i] + 1 << "个点所属面片数量超过10个！" << endl;
		}
		if (pos1>vertexIndices[i + 1] * 20 + 19) {
			pos1 = vertexIndices[i + 1] * 20 + 19;
			cout << "第" << vertexIndices[i + 1] + 1 << "个点所属面片数量超过10个！" << endl;
		}
		if (pos2>vertexIndices[i + 2] * 20 + 19) {
			pos2 = vertexIndices[i + 2] * 20 + 19;
			cout << "第" << vertexIndices[i + 2] + 1 << "个点所属面片数量超过10个！" << endl;
		}
		tri_index[pos0] = i / 3;
		tri_index[pos1] = i / 3;
		tri_index[pos2] = i / 3;
	}
	return true;

}

bool Obj::compute_normal()
{

	vector<glm::vec3> face_normal(vertexIndices.size() / 3, glm::vec3(0, 0, 0));

	for (int i = 0; i<face_normal.size(); i++)
	{
		glm::vec3 pos[3];
		for (int j = 0; j< 3; j++)
			pos[j] = vertices[vertexIndices[i * 3 + j]];

		glm::vec3 side1, side2, normal;
		side1 = pos[1] - pos[0];
		side2 = pos[2] - pos[0];
		face_normal[i] = glm::normalize(glm::cross(side1, side2));
	}

	for (int i = 0; i<vertices.size(); i++)
	{
		normals[i] = glm::vec3(0, 0, 0);
		for (int j = i * 20; tri_index[j] != -1 && j <= i * 20 + 19; j++)
			normals[i] += face_normal[tri_index[j]];

		normals[i] = glm::normalize(normals[i]);
	}
	return true;
}


