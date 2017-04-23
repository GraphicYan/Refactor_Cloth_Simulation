
// 本次代码的重构主要目的在于简化代码的结构，保留最核心的渲染与仿真以及碰撞处理
// 读者可在此基础上很方便的添加各种渲染等
// 另外本次的gpu_kdtree并没有添加回溯搜索
// 考虑到代码的简化，因而未添加仿真停止条件以及最后一次的碰撞检测，所以会有很多穿透
// 读者可自行添加，另外请先阅读readme.txt

// author：TongkuiSu and YanZhang
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

