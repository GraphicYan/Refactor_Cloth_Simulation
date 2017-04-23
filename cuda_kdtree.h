#ifndef __CUDA_KDTREE_H__
#define __CUDA_KDTREE_H__

#include "cpu_kdtree.h"
#include <vector>

struct CUDA_KDNode
{
    int level;
    int parent, left, right;
    float split_value;
    int num_indexes;
    int indexes;
};

using namespace std;

class CUDA_KDTree
{
public:
    ~CUDA_KDTree();
    void CreateKDTree(KDNode *root, int num_nodes, const vector <Point_3> &data);
    void Search(const vector <Point_3> &queries, vector <int> &indexes, vector <float> &dists);

public:
    CUDA_KDNode *m_gpu_nodes;
    int *m_gpu_indexes;
    Point_3 *m_gpu_points;

    int m_num_points;
};

void CheckCUDAError(const char *msg);

#endif
