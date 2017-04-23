#ifndef SIMULATION_H_
#define SIMULATION_H_

#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include "Spring.h"
#include "LoadObj.h"
#include "cuda_kdtree.h"
#include "collision_body.h"

class Simulation
{
public:
	Simulation(Obj& cloth, Springs cloth_springs,Collision_obj& collision_body);
	~Simulation() {}
	void swap_buffer();

	void VerletCUDA(glm::vec4* pos_vbo, const unsigned int num_vertices, const unsigned int cloth_index_size,
		const float & damp, const float & mass, float dt, int stride, bool& finished, const CUDA_KDNode *nodes, const int *indexes, const Point_3 *pts);

private:
	void InitCUDA(const unsigned int cloth_vertices_size,const unsigned int cloth_index_size,
		const unsigned int bodySize, const unsigned int barySize);
	void UploadCUDA(vector<glm::vec4>& positions, vector<Neigh>& neigh, vector<int>& c_index,
		vector<int>& tri_of_point, vector<glm::vec3>& body_position, vector<glm::vec3>& barys);
	void gpu_body_kdtree(vector<glm::vec3>& body_barys);
	
private:
	inline void print_cuda_status(cudaError_t cudaStatus);
	inline void computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads);

public:
	const static unsigned int MAX_NEIGH = 40;  //每个点领域最大个数，包括一级邻域和二级邻域
	const static unsigned int MAX_FACE = 20;   //计算每个点法向量时需要其周围平面

	glm::vec4* X[2];
	glm::vec4* X_last[2];
	glm::vec4 * X_in, *X_out;
	glm::vec4 * X_last_in, *X_last_out;
	int readID, writeID;
	cudaGraphicsResource* cuda_vbo_resource;


	Neigh* cloth_spring_neigh;    //结构弹簧与弯曲弹簧
	glm::vec3* cloth_normal;

	glm::vec3* body_data;       //collision detection
	glm::vec3* bary_centric;    //body barycentric in each face

	int* cloth_index;
	glm::vec3* cloth_face;
	int* g_tri_of_point;        //计算每个点法向量时需要其周围平面

	//whether it is balanced or not
	glm::vec3* vertex_force;
	glm::vec3* vertex_velocity;

	int normalStride;
	int cloth_vertex_size;
	int cloth_index_size;

	//gpu kdtree
	CUDA_KDNode *gpu_nodes;
	int *gpu_indexes;
	Point_3 *gpu_points;
};

#endif