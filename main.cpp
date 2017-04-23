
// ���δ�����ع���ҪĿ�����ڼ򻯴���Ľṹ����������ĵ���Ⱦ������Լ���ײ����
// ���߿��ڴ˻����Ϻܷ������Ӹ�����Ⱦ��
// ���Ȿ�ε�gpu_kdtree��û����ӻ�������
// ���ǵ�����ļ򻯣����δ��ӷ���ֹͣ�����Լ����һ�ε���ײ��⣬���Ի��кܶഩ͸
// ���߿�������ӣ����������Ķ�readme.txt

// author��TongkuiSu and YanZhang
// tongkuisu@smail.nju.edu.cn

#include "stdafx.h"
#include "scene.h"
#include "help.h"
#include "LoadObj.h"
#include "Spring.h"
#include "Simulation.h"
#include "collision_body.h"

int main(int argc, char** argv)
{
	Scene* mainScene = Scene::getInstance(argc, argv); //create an OpenGL render scene 

	Obj body(BODY, "FullBodyWithImage");  
	mainScene->add(body);

	Collision_obj collision_body(body);
	collision_body.scale(1.05);

	Obj cloth(CLOTH,"poloshirt1");
	mainScene->add(cloth);
	Springs cloth_springs(cloth);

	Simulation cloth_simulation(cloth,cloth_springs,collision_body);
	mainScene->add(cloth_simulation);

	mainScene->render();
    return 0;
}

