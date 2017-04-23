#include "stdafx.h"
#include "Simulation.h"
#include "cuda_kdtree.h"
#include <math_functions.h>
#include <iostream> 
#include <cuda_gl_interop.h>
using namespace std;

KDtree cpu_tree;
CUDA_KDTree gpu_tree;

__global__ void face_normal(float4 * g_pos_in, int* cloth_index, const unsigned int cloth_index_size, glm::vec3* cloth_face);   //update cloth face normal
__global__ void verlet(float4 * pos_vbo, float4 * g_pos_in, float4 * g_pos_old_in, float4 * g_pos_out, float4 * g_pos_old_out,
	Neigh* neigh, int MAX_NEIGH, const unsigned int NUM_VERTICES, float damp, float mass, float dt, float3* body, int* face_index, 
	glm::vec3* cloth_face, int stride, glm::vec3* vertex_force, glm::vec3* vertex_velocity, const CUDA_KDNode *nodes, const int *indexes, const Point_3 *pts);  //verlet intergration


Simulation::Simulation(Obj& cloth, Springs cloth_springs, Collision_obj& collision_body) : readID(0), writeID(1)
{
	cudaError_t cudaStatus = cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource,cloth.objVAO.array_buffer, cudaGraphicsMapFlagsWriteDiscard);   	//register vbo
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "register failed\n");

	InitCUDA(cloth.vertices4.size(), cloth.vertexIndices.size(),collision_body.vertices_in_faceseq.size(), collision_body.barys.size());
	normalStride = 4 * cloth.vertices.size() + 2 * cloth.vertices4.size();
	cloth_vertex_size = cloth.vertices4.size();
	cloth_index_size = cloth.vertexIndices.size();
	UploadCUDA(cloth.vertices4,cloth_springs.neighs,cloth.vertexIndices,cloth.tri_index,collision_body.vertices_in_faceseq, collision_body.barys);
}

void Simulation::InitCUDA(const unsigned int cloth_vertices_size, const unsigned int cloth_index_size,
	const unsigned int bodySize, const unsigned int barySize)
{
	size_t heap_size = 128 * 1024 * 1024;  //set heap size to 128M, the default is 8M
	cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size);

	const unsigned int num_threads = cloth_vertices_size;
	const unsigned int num_neigh = cloth_vertices_size*MAX_NEIGH;
	const unsigned int num_tri_of_point = cloth_vertices_size*MAX_FACE;

	const unsigned int vertices_bytes = sizeof(glm::vec4) * num_threads;
	const unsigned int neigh_bytes = sizeof(struct Neigh) * num_neigh;
	const unsigned int normal_bytes = sizeof(glm::vec3) * cloth_vertices_size;
	const unsigned int body_bytes = sizeof(glm::vec3) * bodySize;
	const unsigned int bary_bytes = sizeof(glm::vec3) * barySize;
	const unsigned int cloth_index_bytes = sizeof(int) * cloth_index_size;
	const unsigned int cloth_face_bytes = sizeof(glm::vec3)*cloth_index_size / 3;
	const unsigned int tri_of_point_bytes = sizeof(int)*num_tri_of_point;


	cudaError_t cudaStatus;      	// allocate device memory 
	cudaStatus = cudaMalloc((void**)&X[0], vertices_bytes);			 // cloth vertices
	cudaStatus = cudaMalloc((void**)&X[1], vertices_bytes);			 // cloth vertices
	cudaStatus = cudaMalloc((void**)&X_last[0], vertices_bytes);	 // cloth old vertices
	cudaStatus = cudaMalloc((void**)&X_last[1], vertices_bytes);	 // cloth old vertices

	cudaStatus = cudaMalloc((void**)&cloth_spring_neigh, neigh_bytes);		//cloth springs' neighbour info
	cudaStatus = cudaMalloc((void**)&cloth_normal, normal_bytes);			//cloth normal
	cudaStatus = cudaMalloc((void**)&cloth_index, cloth_index_bytes);		//cloth_index
	cudaStatus = cudaMalloc((void**)&cloth_face, cloth_face_bytes);			//cloth_face
	cudaStatus = cudaMalloc((void**)&g_tri_of_point, tri_of_point_bytes);   //tri_of_point

	cudaStatus = cudaMalloc((void**)&vertex_force, sizeof(glm::vec3)*num_threads);	     // vertex_force
	cudaStatus = cudaMalloc((void**)&vertex_velocity, sizeof(glm::vec3)*num_threads);    //vertex_velocity
	cudaGetLastError();

	cudaStatus = cudaMalloc((void**)&body_data, body_bytes);		 //body vertices in face sequence
	cudaStatus = cudaMalloc((void**)&bary_centric, bary_bytes);      //body barycentrics

	if (cudaStatus != cudaSuccess)
		print_cuda_status(cudaStatus);

}

void Simulation::UploadCUDA(vector<glm::vec4>& cloth_vertices, vector<Neigh>& spring_neigh, vector<int>& c_index,
	vector<int>& tri_of_point, vector<glm::vec3>& body_vertices, vector<glm::vec3>& body_barys)
{
	assert(X[0] != NULL);
	assert(X_last[0] != NULL);

	X_in = X[readID];
	X_out = X[writeID];
	X_last_in = X_last[readID];
	X_last_out = X_last[writeID];



		const int size = cloth_vertices.size();
		const int bodysize = body_vertices.size();
		const int barySize = body_barys.size();
		const unsigned int cloth_index_size = c_index.size();
		const unsigned int num_threads = size;
		const unsigned int vertices_size = sizeof(float4) * num_threads;

		cudaError_t cudaStatus;
		cudaStatus = cudaMemcpy(X_in, &cloth_vertices[0], vertices_size, cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(X_last_in, &cloth_vertices[0], vertices_size, cudaMemcpyHostToDevice);
		cudaStatus = cudaMemcpy(cloth_spring_neigh, &spring_neigh[0], sizeof(Neigh)*num_threads * MAX_NEIGH, cudaMemcpyHostToDevice);  //copy spring_neigh
		cudaStatus = cudaMemcpy(cloth_index, &c_index[0], sizeof(int)*cloth_index_size, cudaMemcpyHostToDevice);				//copy cloth_index
		cudaStatus = cudaMemcpy(g_tri_of_point, &tri_of_point[0], sizeof(int)*size * MAX_FACE, cudaMemcpyHostToDevice);		    //copy tri_of_point
		cudaStatus = cudaMemcpy(body_data, &body_vertices[0], sizeof(glm::vec3)*bodysize, cudaMemcpyHostToDevice);					//copy body
		cudaStatus = cudaMemcpy(bary_centric, &body_barys[0], sizeof(glm::vec3)*barySize, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
			print_cuda_status(cudaStatus);

		//create static kdtree for body in gpu
		gpu_body_kdtree(body_barys);

	int tmp = readID;
	readID = writeID;
	writeID = tmp;

}

void Simulation::gpu_body_kdtree(vector<glm::vec3>& body_barys)
{

	int max_tree_levels = 10; // play around with this value to get the best result
	vector <Point_3> data;
	for (auto vertex : body_barys)
	{
		Point_3 tem;
		tem.coords[0] = vertex.x;
		tem.coords[1] = vertex.y;
		tem.coords[2] = vertex.z;
		data.push_back(tem);
	}
	cpu_tree.Create(data, max_tree_levels);
	gpu_tree.CreateKDTree(cpu_tree.GetRoot(), cpu_tree.GetNumNodes(), data);

	//save the pointer
	gpu_nodes = gpu_tree.m_gpu_nodes;
	gpu_indexes = gpu_tree.m_gpu_indexes;
	gpu_points = gpu_tree.m_gpu_points;
	cout << "gpu kdtree build successfully!" << endl;
}

void Simulation::swap_buffer()
{
	X_in = X[readID];
	X_out = X[writeID];
	X_last_in = X_last[readID];
	X_last_out = X_last[writeID];

	int tmp = readID;
	readID = writeID;
	writeID = tmp;
}

void Simulation::VerletCUDA(glm::vec4* pos_vbo, const unsigned int num_vertices, const unsigned int cloth_index_size,
	const float & damp, const float & mass, float dt, int stride, bool& finished, const CUDA_KDNode *nodes, const int *indexes, const Point_3 *pts)
{
	cudaError_t cudaStatus;
	unsigned int numThreads0, numBlocks0;
	computeGridSize(cloth_index_size / 3, 512, numBlocks0, numThreads0);
	face_normal <<<numBlocks0, numThreads0 >>>((float4*)X_in, cloth_index, cloth_index_size, cloth_face);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess)
		fprintf(stderr, "normal cudaDeviceSynchronize returned error code %d after launching addKernel!\n%s\n", cudaStatus, cudaGetErrorString(cudaStatus));

	// setup execution parameters 
	unsigned int numThreads, numBlocks;
	unsigned int numParticles = num_vertices;

	computeGridSize(numParticles, 512, numBlocks, numThreads);
	verlet <<< numBlocks, numThreads >>>((float4*)pos_vbo, (float4*)X_in, (float4*)X_last_in, (float4*)X_out, (float4*)X_last_out, cloth_spring_neigh, MAX_NEIGH, num_vertices, damp,
		mass, dt, (float3*)body_data, g_tri_of_point, cloth_face, stride, vertex_force, vertex_velocity, gpu_nodes,gpu_indexes,gpu_points);

	// stop the CPU until the kernel has been executed
	cudaStatus = cudaDeviceSynchronize();
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n%s\n",
			cudaStatus, cudaGetErrorString(cudaStatus));
		system("pause");
	}
		
}

void Simulation::print_cuda_status(cudaError_t cudaStatus)
{
	printf("Cuda error: %s.\n", cudaGetErrorString(cudaStatus));
}

void Simulation::computeGridSize(unsigned int n, unsigned int blockSize, unsigned int &numBlocks, unsigned int &numThreads)
{
	numThreads = min(blockSize, n);
	numBlocks = (n % numThreads != 0) ? (n / numThreads + 1) : (n / numThreads);
}