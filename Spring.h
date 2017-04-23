#pragma once

#include "stdafx.h"
#include "LoadObj.h"

#define VOLUME 40    //����ÿ�������洢���������
#define THIGHTNESS 40  //�������ϵ��
#define COEFFICIENT 0.5f  //����ϵ��Ԫ

enum ELASTICITY { RIGIDITY, NORMAL, BEND, BOUNDARY };

const float KS[4] = { 100.0F*COEFFICIENT,30.0F*COEFFICIENT,30.0F*COEFFICIENT,80.0F*COEFFICIENT };


struct Neigh
{
	int neigh_index;
	bool neigh_type;
	ELASTICITY elasticity_type;
	float original_length;
};


class Matrix
{
public:
	Matrix() {};
	~Matrix();
	bool Insert_Matrix(int i, int j, int k, std::vector<int> &value_inline);
	int Get_Size();
	int Get_point1(int n);
	int Get_point2(int n);
private:
	std::vector<int>  column;
	std::vector<int>  row;
	std::vector<int>  value;
};


class Springs
{
public:
	Springs(Obj& cloth_info);
	~Springs() {};
	vector<Neigh> neighs;
	bool add_spring(int a, int b, ELASTICITY e_type, bool n_type);
	Obj *cloth;

private:
	bool normal_spring();
	bool piece_boundary_spring();
	bool boundary_boundary_spring();
	bool bend_spring();

};
