#include <cuda.h>
#include <device_functions.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <glm/glm.hpp>
#include "cuda_kdtree.h"

#include "Spring.h"

struct kdres    //the nearest point
{
	float pos[3];
	int resIndex;
};

const __device__ float device_ks[4] = { 50.0,15.0,15.0,4.0};
const __device__ float	nearestDis = 0.008;

#define CUDA_STACK 100 // fixed size stack elements for each thread, increase as required. Used in SearchAtNodeRange_2.
__device__ float Distance_2(const Point_3 &a, const Point_3 &b)
{
	float dist = 0;

	for (int i = 0; i < KDTREE_DIM; i++) {
		float d = a.coords[i] - b.coords[i];
		dist += d*d;
	}

	return dist;
}

__device__ void SearchAtNode_2(const CUDA_KDNode *nodes, const int *indexes, const Point_3 *pts, int cur, const Point_3 &query, int *ret_index, float *ret_dist, int *ret_node)
{
	// Finds the first potential candidate

	int best_idx = 0;
	float best_dist = FLT_MAX;

	while (true) {
		int split_axis = nodes[cur].level % KDTREE_DIM;

		if (nodes[cur].left == -1) {
			*ret_node = cur;

			for (int i = 0; i < nodes[cur].num_indexes; i++) {
				int idx = indexes[nodes[cur].indexes + i];
				float dist = Distance_2(query, pts[idx]);
				if (dist < best_dist) {
					best_dist = dist;
					best_idx = idx;
				}
			}

			break;
		}
		else if (query.coords[split_axis] < nodes[cur].split_value) {
			cur = nodes[cur].left;
		}
		else {
			cur = nodes[cur].right;
		}
	}

	*ret_index = best_idx;
	*ret_dist = best_dist;
}

__device__ void SearchAtNodeRange_2(const CUDA_KDNode *nodes, const int *indexes, const Point_3 *pts, const Point_3 &query, int cur, float range, int *ret_index, float *ret_dist)
{
	// Goes through all the nodes that are within "range"

	int best_idx = 0;
	float best_dist = FLT_MAX;

	// Ok, we don't have nice STL vectors to use, and we can't dynamically allocate memory with CUDA??
	// We'll use a fixed length stack, increase this as required
	int to_visit[CUDA_STACK];
	int to_visit_pos = 0;

	to_visit[to_visit_pos++] = cur;

	while (to_visit_pos) {
		int next_search[CUDA_STACK];
		int next_search_pos = 0;

		while (to_visit_pos) {
			cur = to_visit[to_visit_pos - 1];
			to_visit_pos--;

			int split_axis = nodes[cur].level % KDTREE_DIM;

			if (nodes[cur].left == -1) {
				for (int i = 0; i < nodes[cur].num_indexes; i++) {
					int idx = indexes[nodes[cur].indexes + i];
					float d = Distance_2(query, pts[idx]);

					if (d < best_dist) {
						best_dist = d;
						best_idx = idx;
					}
				}
			}
			else {
				float d = query.coords[split_axis] - nodes[cur].split_value;

				// There are 3 possible scenarios
				// The hypercircle only intersects the left region
				// The hypercircle only intersects the right region
				// The hypercricle intersects both

				if (fabs(d) > range) {
					if (d < 0)
						next_search[next_search_pos++] = nodes[cur].left;
					else
						next_search[next_search_pos++] = nodes[cur].right;
				}
				else {
					next_search[next_search_pos++] = nodes[cur].left;
					next_search[next_search_pos++] = nodes[cur].right;
				}
			}
		}

		// No memcpy available??
		for (int i = 0; i < next_search_pos; i++)
			to_visit[i] = next_search[i];

		to_visit_pos = next_search_pos;
	}

	*ret_index = best_idx;
	*ret_dist = best_dist;
}
// i can use search which calls searchatnode and searchatnoderange(backtracking)
__device__ void Search_2(const CUDA_KDNode *nodes, const int *indexes, const Point_3 *pts, const Point_3 &query, int *ret_index, float *ret_dist)
{
	// Find the first closest node, this will be the upper bound for the next searches
	int best_node = 0;
	int best_idx = 0;
	float best_dist = FLT_MAX;
	float radius = 0;

	SearchAtNode_2(nodes, indexes, pts, 0 /* root */, query, &best_idx, &best_dist, &best_node);

	//radius = sqrt(best_dist);

	//// Now find other possible candidates
	//int cur = best_node;

	//while(nodes[cur].parent != -1) {
	//    // Go up
	//    int parent = nodes[cur].parent;
	//    int split_axis = nodes[parent].level % KDTREE_DIM;

	//    // Search the other node
	//    float tmp_dist = FLT_MAX;
	//    int tmp_idx;

	//    if(fabs(nodes[parent].split_value - query.coords[split_axis]) <= radius) {
	//        // Search opposite node
	//       if(nodes[parent].left != cur)
	//            SearchAtNodeRange_2(nodes, indexes, pts, query, nodes[parent].left, radius, &tmp_idx, &tmp_dist);
	//        else
	//            SearchAtNodeRange_2(nodes, indexes, pts, query, nodes[parent].right, radius, &tmp_idx, &tmp_dist);
	//    }

	//    if(tmp_dist < best_dist) {
	//        best_dist = tmp_dist;
	//        best_idx = tmp_idx;
	//    }

	//    cur = parent;
	//}

	*ret_index = best_idx;
	*ret_dist = best_dist;
}


__device__ bool outside(const glm::vec3* pri, const glm::vec3 p, float &dist, glm::vec3 &normal)
{
	glm::vec3 side1, side2, normalface;
	side1 = pri[1] - pri[0];
	side2 = pri[2] - pri[0];
	normalface = glm::cross(side1, side2);
	normal = glm::normalize(normalface);

	glm::vec3 tem = p - pri[0];
	dist = glm::dot(tem, normal);
	if (dist > 0)
		return true;
	else
	{
		dist = -1 * dist;
		return false;
	}


}

__device__ bool collisionResponse(glm::vec3& pos, glm::vec3& force, glm::vec3& pos_old, float3* body, const CUDA_KDNode *nodes, const int *indexes, const Point_3 *pts)
{
	Point_3 query;
	query.coords[0] = pos.x;   query.coords[1] = pos.y;   query.coords[2] = pos.z;
	int ret_index;
	float ret_dist;
	Search_2(nodes, indexes, pts, query, &ret_index, &ret_dist);

	kdres result;
	float length = ret_dist;
	result.resIndex = ret_index;

	if (length < nearestDis) // 当点靠近时再判断是否在外面（如果太远显然在外）
	{

		glm::vec3 primitive[3];
		int idx = result.resIndex * 3;
		for (int i = 0; i < 3; i++)
		{
			primitive[i] = glm::vec3(body[idx + i].x, body[idx + i].y, body[idx + i].z);
		}

		float dist;
		glm::vec3 normal;
		if (!outside(primitive, pos, dist, normal))
		{
			dist = 8.0f*dist;    //原为5.5
			glm::vec3 temp = -dist*normal;
			force = force - temp;

			float ratio = 3.0 / 3.0;
			glm::vec3 temppos = pos*ratio;
			glm::vec3 temppos_old = pos_old*(1 - ratio);
			pos_old = temppos + temppos_old;
			return true;

		}
	}
	return false;

}

__global__ void face_normal(float4 * g_pos_in, int* cloth_index, const unsigned int cloth_index_size, glm::vec3* cloth_face)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int max_thread = cloth_index_size / 3;
	if (index >= max_thread)
		return;

	unsigned int f_index[3];
	for (int i = 0; i<3; i++)
		f_index[i] = index * 3 + i;

	float4 vertex[3];
	for (int i = 0; i < 3; i++)
		vertex[i] = g_pos_in[cloth_index[f_index[i]]];  //find the fucking bug!

	glm::vec3 pos[3];
	for (int i = 0; i < 3; i++)
		pos[i] = glm::vec3(vertex[i].x, vertex[i].y, vertex[i].z);

	glm::vec3 side1, side2, normal;
	side1 = pos[1] - pos[0];
	side2 = pos[2] - pos[0];
	normal = glm::normalize(glm::cross(side1, side2));

	cloth_face[index] = normal;

}


__global__ void verlet(float4 * pos_vbo, float4 * g_pos_in, float4 * g_pos_old_in, float4 * g_pos_out, float4 * g_pos_old_out,
	Neigh* neigh, int MAX_NEIGH, const unsigned int NUM_VERTICES, float damp, float mass, float dt, float3* body, int* face_index, glm::vec3* cloth_face, int stride, 
	glm::vec3* vertex_force, glm::vec3* vertex_velocity, const CUDA_KDNode *nodes, const int *indexes, const Point_3 *pts)
{
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index >= NUM_VERTICES)
		return;
	float ks = 0, kd = 0;

	volatile float4 posData = g_pos_in[index];
	volatile float4 posOldData = g_pos_old_in[index];


	glm::vec3 pos = glm::vec3(posData.x, posData.y, posData.z);
	glm::vec3 pos_old = glm::vec3(posOldData.x, posOldData.y, posOldData.z);
	glm::vec3 vel = (pos - pos_old) / dt;
	vertex_velocity[index] = vel;//get the velocity

	const glm::vec3 gravity = glm::vec3(0.0f, -0.000981*2.0f, 0.0f); //set gravity
	glm::vec3 force = gravity*mass + vel*damp;

	int first_neigh = index*MAX_NEIGH;
	int time = 0;
	for (int k = first_neigh; neigh[k].neigh_index != -1 && time<MAX_NEIGH; k++, time++) //部分点邻域大于MAX_NEIGH(20)
	{


		if (neigh[k].elasticity_type == RIGIDITY)  ks = device_ks[0];
		if (neigh[k].elasticity_type == NORMAL)  ks = device_ks[1];
		if (neigh[k].elasticity_type == BEND)  ks = device_ks[2];
		if (neigh[k].elasticity_type == BOUNDARY)  ks = device_ks[3];

		int index_neigh = neigh[k].neigh_index;
		volatile float4 pos_neighData = g_pos_in[index_neigh];
		volatile float4 pos_lastData = g_pos_old_in[index_neigh];
		glm::vec3 p2 = glm::vec3(pos_neighData.x, pos_neighData.y, pos_neighData.z);
		glm::vec3 p2_last = glm::vec3(pos_lastData.x, pos_lastData.y, pos_lastData.z);

		glm::vec3 v2 = (p2 - p2_last) / dt;
		glm::vec3 deltaP = pos - p2;
		if (glm::length(deltaP) == 0) { force += glm::vec3(0.0f); continue; }//deltaP += glm::vec3(0.0001);	//avoid '0'

		glm::vec3 deltaV = vel - v2;
		float dist = glm::length(deltaP); //avoid '0'


		float original_length = neigh[k].original_length;
		float leftTerm = -ks * (dist - original_length);
		float  rightTerm = kd * (glm::dot(deltaV, deltaP) / dist);
		glm::vec3 springForce = (leftTerm + rightTerm)*glm::normalize(deltaP);

		force += springForce;

	}

	collisionResponse(pos, force, pos_old, body, nodes, indexes, pts);              //add collision detection
	vertex_force[index] = force;  //get the vertex_force
	glm::vec3 acc = force / mass;

	glm::vec3 tmp = pos;
	pos = pos + pos - pos_old + acc * dt * dt;
	pos_old = tmp;

	//compute point normal
	glm::vec3 normal(0.0);
	int first_face_index = index * 20;
	for (int i = first_face_index, time = 0; face_index[i] != -1 && time<20; i++, time++)
	{
		int findex = face_index[i];
		glm::vec3 fnormal = cloth_face[findex]; //volatile?
		normal += fnormal;
	}
	normal = glm::normalize(normal);
	pos_vbo[index] = make_float4(pos.x, pos.y, pos.z, posData.w);
	float3 *normalPos = (float3*)((float*)pos_vbo + stride); //use X_last to compute normal, small error
	normalPos[index] = make_float3(normal.x, normal.y, normal.z);

	g_pos_out[index] = make_float4(pos.x, pos.y, pos.z, posData.w);
	g_pos_old_out[index] = make_float4(pos_old.x, pos_old.y, pos_old.z, posOldData.w);


}
