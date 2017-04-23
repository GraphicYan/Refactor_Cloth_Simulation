#include "stdafx.h"
#include "Spring.h"
#include <iostream>
#include "kdtree.h"

#define REGARDLESS 0
#define FIRST_NEIGH 0
#define SECOND_NEIGH 1


vector<int> boundary_num;        //为加载边界缝合弹簧时方便，定义此两个向量。
vector<int> vertex_in_boundary;  //这两个向量的赋值在piece_boundary_spring中，使用在boundary_boundary_spring中





Matrix::~Matrix()
{
	row.clear();
	column.clear();
	value.clear();
}


bool Matrix::Insert_Matrix(
	int i,
	int j,
	int k,
	std::vector<int> &value_inline)
{
	for (int n = 0; n<row.size(); n++)
	{
		if ((row[n] == i&&column[n] == j) || (row[n] == j&&column[n] == i)) {
			value_inline.push_back(value[n]);
			value_inline.push_back(k);
			return 0;
		}
	}
	row.push_back(i);
	column.push_back(j);
	value.push_back(k);
	return 1;
}


int Matrix::Get_Size()
{
	return row.size();
}


int Matrix::Get_point1(int n) {
	return row[n];
}

int Matrix::Get_point2(int n) {
	return column[n];
}



bool Springs::add_spring(int a, int b, ELASTICITY e_type, bool n_type)
{
	int pos_a = a*VOLUME;
	int pos_b = b*VOLUME;

	while (neighs[pos_a].neigh_index != -1) {
		pos_a++;
	}

	while (neighs[pos_b].neigh_index != -1) {
		pos_b++;
	}

	if (pos_a - a*VOLUME >= VOLUME) {

		cout << "第" << a << "个点" << "弹簧数目超限！" << endl;
		return 0;
	}
	if (pos_b - b*VOLUME >= VOLUME) {

		cout << "第" << b << "个点" << "弹簧数目超限！" << endl;
		return 0;
	}

	glm::vec3 deltaP = (*cloth).vertices[a] - (*cloth).vertices[b];

	neighs[pos_a].neigh_index = b;
	neighs[pos_a].elasticity_type = e_type;
	neighs[pos_a].neigh_type = n_type;

	neighs[pos_b].neigh_index = a;
	neighs[pos_b].elasticity_type = e_type;
	neighs[pos_b].neigh_type = n_type;

	if (e_type == BOUNDARY) {
		neighs[pos_a].original_length = 0.0f;
		neighs[pos_b].original_length = 0.0f;
	}

	else if ( e_type == RIGIDITY) {
		neighs[pos_a].original_length = sqrt(glm::dot(deltaP, deltaP));
		neighs[pos_b].original_length = sqrt(glm::dot(deltaP, deltaP));
	}

	else {
		neighs[pos_a].original_length = sqrt(glm::dot(deltaP, deltaP))*THIGHTNESS / (THIGHTNESS + 1);
		neighs[pos_b].original_length = sqrt(glm::dot(deltaP, deltaP))*THIGHTNESS / (THIGHTNESS + 1);
	}
	return 1;
}


bool Springs::normal_spring()
{
	for (int i = 0; i<(*cloth).groups.size(); i++)
	{

		if ((*cloth).groups[i].IF_ELASTICITY == 1)   //若是弹性部分
			for (int n = (*cloth).groups[i].getFirstTri(); n <= (*cloth).groups[i].getLastTri(); n += 3)
			{
				add_spring((*cloth).vertexIndices[n],
					(*cloth).vertexIndices[n + 1],
					NORMAL, FIRST_NEIGH);
				add_spring((*cloth).vertexIndices[n],
					(*cloth).vertexIndices[n + 2],
					NORMAL, FIRST_NEIGH);
				add_spring((*cloth).vertexIndices[n + 1],
					(*cloth).vertexIndices[n + 2],
					NORMAL, FIRST_NEIGH);
			}
		else                                  //若是非弹性部分
			for (int n = (*cloth).groups[i].getFirstTri(); n <= (*cloth).groups[i].getLastTri(); n += 3)
			{
				add_spring((*cloth).vertexIndices[n],
					(*cloth).vertexIndices[n + 1],
					RIGIDITY, FIRST_NEIGH);
				add_spring((*cloth).vertexIndices[n],
					(*cloth).vertexIndices[n + 2],
					RIGIDITY, FIRST_NEIGH);
				add_spring((*cloth).vertexIndices[n + 1],
					(*cloth).vertexIndices[n + 2],
					RIGIDITY, FIRST_NEIGH);
			}
	}
	return 1;
}


bool Springs::piece_boundary_spring()
{
	for (int n = 0; n<(*cloth).groups.size() - 3 - REGARDLESS; n += 3)
	{

		kdtree *kd = kd_create(3);
		int *idx = new int[(*cloth).groups[n].size()];


		for (int i = (*cloth).groups[n].gethead(); i <= (*cloth).groups[n].getrear(); i++)   //为面片1建立kdtree
		{
			idx[i - (*cloth).groups[n].gethead()] = i;
			int ret = kd_insert3f(kd, (*cloth).vertices[i].x,
				(*cloth).vertices[i].y,
				(*cloth).vertices[i].z,
				&idx[i - (*cloth).groups[n].gethead()]);
		}

		for (int i = (*cloth).groups[n + 1].gethead(); i <= (*cloth).groups[n + 1].gethead(); i++)   //将面片2的点全部放到范围之外
		{
			(*cloth).vertices[i].x = (*cloth).vertices[i].y = (*cloth).vertices[i].z = -100.0f;

		}

		for (int i = (*cloth).groups[n + 2].gethead(); i <= (*cloth).groups[n + 2].getrear(); i++)    //为边界中的点找最邻近点
		{
			float kdpos[3];
			kdres *result = kd_nearest3f(kd, (*cloth).vertices[i].x,
				(*cloth).vertices[i].y,
				(*cloth).vertices[i].z);
			int *resultidx = (int*)kd_res_itemf(result, kdpos);

			add_spring(resultidx[0], i, NORMAL, SECOND_NEIGH);

			vertex_in_boundary.push_back(i);
		}
		boundary_num.push_back(vertex_in_boundary.size());

		kd_free(kd);

		delete[]idx;

	}
	return true;
}


bool Springs::boundary_boundary_spring()
{
	///////////////////////////////////计算边界缝合线与面片之间点的最大距离/////////////////////
	float Max_dist = 0;
	for (int i = 0; i<neighs.size(); i++)
	{
		if (neighs[i].original_length>Max_dist)
			Max_dist = neighs[i].original_length;

	}
	cout << "边界与面片最邻点之间的最大距离：" << Max_dist << endl;

	//////////////////////////////////为边界之间建立弹簧////////////////////////////////////////////////
	for (int n = 0; n<boundary_num.size() - 1; n++)
	{
		kdtree *BoundaryKD = kd_create(3);
		int * Boundary_idx = new int[boundary_num.back() - boundary_num[n + 1] + boundary_num[n]];
		for (int i = 0; i<boundary_num[n]; i++)
		{
			int num = vertex_in_boundary[i];
			Boundary_idx[i] = num;
			int ret = kd_insert3f(BoundaryKD, (*cloth).vertices[num].x,
				(*cloth).vertices[num].y,
				(*cloth).vertices[num].z, &Boundary_idx[i]);
		}
		for (int i = boundary_num[n + 1]; i<boundary_num.back(); i++)
		{
			int num = vertex_in_boundary[i];
			Boundary_idx[i - boundary_num[n + 1] + boundary_num[n]] = num;
			int ret = kd_insert3f(BoundaryKD, (*cloth).vertices[num].x,
				(*cloth).vertices[num].y,
				(*cloth).vertices[num].z, &Boundary_idx[i - boundary_num[n + 1] + boundary_num[n]]);
		}

		for (int i = boundary_num[n]; i<boundary_num[n + 1]; i++)
		{
			int num = vertex_in_boundary[i];
			float kdpos[3];
			kdres *result = kd_nearest3f(BoundaryKD, (*cloth).vertices[num].x,
				(*cloth).vertices[num].y,
				(*cloth).vertices[num].z);
			int *resultidx = (int*)kd_res_itemf(result, kdpos);

			if (glm::distance((*cloth).vertices[vertex_in_boundary[i]], (*cloth).vertices[resultidx[0]]) <= Max_dist * 10)////加入距离判断，防止错连
			{

				add_spring(vertex_in_boundary[i], resultidx[0], BOUNDARY, SECOND_NEIGH);

			}


		}
		kd_free(BoundaryKD);
		delete[]Boundary_idx;
	}
	return 1;
}


bool Springs::bend_spring()
{
	Matrix NR;   //Neighbour Relation
	vector<int> point_inline;  //存储两个共边三角形对角顶点索引

	for (int n = 0; n<(*cloth).vertexIndices.size(); n += 3)
	{
		NR.Insert_Matrix((*cloth).vertexIndices[n],
			(*cloth).vertexIndices[n + 1],
			(*cloth).vertexIndices[n + 2], point_inline);
		NR.Insert_Matrix((*cloth).vertexIndices[n],
			(*cloth).vertexIndices[n + 2],
			(*cloth).vertexIndices[n + 1], point_inline);
		NR.Insert_Matrix((*cloth).vertexIndices[n + 1],
			(*cloth).vertexIndices[n + 2],
			(*cloth).vertexIndices[n], point_inline);
	}


	//////////////////////////////////添加弯曲弹簧/////////////////////////////////
	cout << "弯曲弹簧数量：" << point_inline.size() / 2 << endl;
	for (int i = 0; i<point_inline.size(); i += 2) {
		add_spring(point_inline[i], point_inline[i + 1], BEND, SECOND_NEIGH);
	}
	return 1;
}



Springs::Springs(Obj& cloth_info)
{
	cloth = &cloth_info;

	Neigh new_neigh;
	for (int i = 0; i<(*cloth).size()*VOLUME; i++)
	{
		new_neigh.neigh_index = -1;
		neighs.push_back(new_neigh);
	};


	piece_boundary_spring();
	boundary_boundary_spring();
	normal_spring();
	//bend_spring();
}