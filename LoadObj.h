#pragma once
#include "help.h"
#include <glm/glm.hpp>
#include <GL/glew.h>
#include <vector>
using namespace std;

enum  ObjType { CLOTH, BODY };

struct Material
{
	char mtl_name[128];
	char text_path[128];
	GLuint clothTexture;
	float Ns;
	float Ni;
	float d;
	float Tr;
	int illum;
	glm::vec3 Tf;
	glm::vec3 Ka;
	glm::vec3 Kd;
	glm::vec3 Ks;
	glm::vec3 Ke;
};

class Group
{
public:
	Group(int f, int l, Material mtl);
	~Group() { };
	int gethead() { return First; };
	int getrear() { return Last; };
	int getFirstTri() { return First_Tri; };
	int getLastTri() { return Last_Tri; };
	bool set_tri(int first_tri, int last_tri);
	int size() { return Group_size; };
	Material getmtl() { return Group_Material; };
	bool IF_ELASTICITY;

public:
	int First;
	int Last;
	int First_Tri;
	int Last_Tri;
	Material Group_Material;
	int Group_size;

};

class VAO
{
public:
	VAO() :vao(0), array_buffer(0), index_buffer(0) {}
	VAO(GLuint _vao, GLuint _array_buffer, GLuint _index_buffer)
		:vao(_vao), array_buffer(_array_buffer), index_buffer(_index_buffer) {}
public:
	GLuint vao;
	GLuint array_buffer;
	GLuint index_buffer;
};

class Obj
{
public:
	Obj() {}
	Obj(ObjType objType, const char * clothpath);
	~Obj() {}
	bool load(ObjType objType,const char * clothpath);
	int size() { return vertices.size(); }

public:
	ObjType obj_type;
	VAO objVAO;   //bind to vao and buffer
	vector<glm::vec4> vertices4;
	vector<glm::vec3> vertices;
	vector<glm::vec3> normals;
	vector<glm::vec2> uvs;
	vector<int> vertexIndices;
	vector<Group> groups;
	vector<int> tri_index; //存储每个点所在的所有面索引


private:
	vector<int> uvIndices;
	vector<int> normalIndices;

private:
	bool loadOBJ_cloth(const char * clothpath);
	bool loadOBJ_body(const char * bodypath);
	bool change_size(float scale,float x,float y,float z);
	bool uv_normalize();
	bool normal_normalize();
	bool count_tri();    //用于计算每个Group包含的三角面的首、尾索引
	void vec3_to_vec4(vector<glm::vec3>& old_vertices, vector<glm::vec4>& new_vertices);
	bool set_tri_index();
	bool compute_normal();
	bool RemoveBack();
};


bool read_mtl(char *mtlpath, vector<Material> &all_mtl);
bool find_mtl(char *mtl_name, vector<Material> all_mtl, Material &mtl);
bool bind_mtl(char *mtl_path, GLuint& clothTexture);




