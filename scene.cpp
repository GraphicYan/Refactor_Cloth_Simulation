#include "stdafx.h"
#include "scene.h"
#include "help.h"
#include <GL/freeglut.h>
#include <GL/wglew.h>
#include <iostream>
using namespace std;


// OPENGL场景的各种参数declaration
Scene* Scene::pscene = nullptr;
int Scene::oldX = 0, Scene::oldY = 0;
float Scene::rX = 15, Scene::rY = 0;
int Scene::state = 1;
float Scene::dist = -23;
float Scene::dy = 0;
GLint Scene::viewport[4];
GLfloat Scene::modelview[16];
GLfloat Scene::projection[16];
glm::vec3 Scene::Up = glm::vec3(0, 1, 0),
		  Scene::Right = glm::vec3(0, 0, 0), 
		  Scene::viewDir= glm::vec3(0, 0, 0);
int Scene::selected_index = -1;


Scene* Scene::getInstance(int argc, char** argv)
{
	if (pscene == nullptr)
		pscene = new Scene(argc,argv);

	return pscene;
}
Scene::Scene(int argc, char** argv):p_simulation(nullptr)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(width,height);
	glutCreateWindow("GLUT Cloth Demo");
	
	GLenum err = glewInit();
	if (err != GLEW_OK) {
		fprintf(stderr, "%s\n", glewGetErrorString(err));
		return;
	}
	wglSwapIntervalEXT(0);  // disable Vertical synchronization

}
void Scene::render()
{
	loadShader(); //InitGL(); //load shader

	glutDisplayFunc(onRender);
	glutReshapeFunc(OnReshape);
	glutIdleFunc(OnIdle);

	glutMouseFunc(OnMouseDown);
	glutMotionFunc(OnMouseMove);
	glutKeyboardFunc(OnKey);
	glutCloseFunc(OnShutdown);

	glutMainLoop();
}

void Scene::RenderBuffer(Obj& object)
{
	GLfloat eyeDir[3] = { viewDir.x,viewDir.y,viewDir.z };

	renderShader.Use();
	glUniformMatrix4fv(renderShader("modelview"), 1, GL_FALSE, modelview);   // the platform does not support "glUniformMatrix4dv"
	glUniformMatrix4fv(renderShader("projection"), 1, GL_FALSE, projection);
	glUniform3fv(renderShader("viewPos"), 1, eyeDir);

	glBindVertexArray(object.objVAO.vao);
	
	GLint* ptr = (GLint*)glMapBuffer(GL_ELEMENT_ARRAY_BUFFER, GL_READ_ONLY);
	glUnmapBuffer(GL_ELEMENT_ARRAY_BUFFER); // unmap it after use

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, object.objVAO.index_buffer);
	for (int i = 0; i < object.groups.size(); i++)
	{
		unsigned int group_size = object.groups[i].Last_Tri - object.groups[i].First_Tri + 1;
		GLuint start = object.groups[i].First_Tri;
		glBindTexture(GL_TEXTURE_2D, object.groups[i].Group_Material.clothTexture);
		glDrawElements(GL_TRIANGLES, (GLsizei)group_size, GL_UNSIGNED_INT, (void*)&ptr[start]);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindVertexArray(0);	

	renderShader.UnUse();
}
void Scene::add(Obj& object)
{
	objs.push_back(&object);
	//add VAOs and Buffers
	GLuint vao;
	GLuint array_buffer;
	GLuint index_buffer;
	
	glGenVertexArrays(1, &vao);
	glGenBuffers(1,&array_buffer);
	glGenBuffers(1,&index_buffer);
	check_GL_error();

	object.objVAO.vao = vao;
	object.objVAO.array_buffer = array_buffer;
	object.objVAO.index_buffer = index_buffer;  
	
	glBindVertexArray(object.objVAO.vao);
	glBindBuffer(GL_ARRAY_BUFFER, object.objVAO.array_buffer);

	glBufferData(GL_ARRAY_BUFFER, sizeof(glm::vec4)*object.vertices4.size() + sizeof(glm::vec2)*object.uvs.size() + sizeof(glm::vec3)*object.normals.size(), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(glm::vec4)*object.vertices4.size(), &object.vertices4[0]);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::vec4)*object.vertices4.size(), sizeof(glm::vec2)*object.uvs.size(), &object.uvs[0]);
	glBufferSubData(GL_ARRAY_BUFFER, sizeof(glm::vec4)*object.vertices4.size() + sizeof(glm::vec2)*object.uvs.size(), sizeof(glm::vec3)*object.normals.size(), &object.normals[0]);
	check_GL_error();

	glVertexAttribPointer(position, 4, GL_FLOAT, GL_FALSE, sizeof(glm::vec4), 0);
	glVertexAttribPointer(texture, 2, GL_FLOAT, GL_FALSE, sizeof(glm::vec2), (const GLvoid*)(sizeof(glm::vec4)*object.vertices4.size()));
	glVertexAttribPointer(normal, 3, GL_FLOAT, GL_FALSE, sizeof(glm::vec3), (const GLvoid*)(sizeof(glm::vec4)*object.vertices4.size() + sizeof(glm::vec2)*object.uvs.size()));

	glEnableVertexAttribArray(position);
	glEnableVertexAttribArray(texture);
	glEnableVertexAttribArray(normal);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,index_buffer);  
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLuint)*object.vertexIndices.size(), &object.vertexIndices[0], GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glBindVertexArray(0);

}
void Scene::add(Simulation& simulation)
{
	p_simulation = &simulation;
}
void Scene::check_GL_error()
{
	assert(glGetError() == GL_NO_ERROR);
}
void Scene::loadShader()
{
	//set light
	GLfloat lightPos[3] = { 0,0.0f,10.0f };
	GLfloat lightColor[3] = { 0.8,0.8,0.8 };
	GLfloat objectColor[3] = { 0.8,0.8,0.8 };
	check_GL_error();

	renderShader.LoadFromFile(GL_VERTEX_SHADER, "shaders/render.vert");
	renderShader.LoadFromFile(GL_FRAGMENT_SHADER, "shaders/render.frag");
	renderShader.CreateAndLinkProgram();

	renderShader.Use();
		renderShader.AddUniform("color");
		renderShader.AddUniform("modelview");
		renderShader.AddUniform("projection");
		renderShader.AddUniform("lightPos");                 glUniform3fv(renderShader("lightPos"), 1, lightPos);
		renderShader.AddUniform("viewPos");
		renderShader.AddUniform("lightColor");               glUniform3fv(renderShader("lightColor"), 1, lightColor);
		renderShader.AddUniform("objectColor");              glUniform3fv(renderShader("objectColor"), 1, objectColor);
		//renderShader.AddUniform("ourTexture");               glUniform1i(renderShader("ourTexture"), 0);
	renderShader.UnUse();
	check_GL_error();

	glEnable(GL_DEPTH_TEST);  
}
Scene::~Scene()
{

}


// OPENGL场景的各种函数
void Scene::DrawGrid()
{
	const int GRID_SIZE = 10;
	glBegin(GL_LINES);
	glColor3f(0.5f, 0.5f, 0.5f);
	for (int i = -GRID_SIZE; i <= GRID_SIZE; i++)
	{
		glVertex3f((float)i, -2, (float)-GRID_SIZE);
		glVertex3f((float)i, -2, (float)GRID_SIZE);

		glVertex3f((float)-GRID_SIZE, -2, (float)i);
		glVertex3f((float)GRID_SIZE, -2, (float)i);
	}

	glEnd();

}
void Scene::RenderGPU_CUDA()
{
	//set simulation parameters
	const float DEFAULT_DAMPING = -0.0125f;
	float mass = 0.3f;
	float timeStep = 1.0f / 50.0f;
	bool finished;

	Simulation* p_simulate = pscene->p_simulation;
	if (p_simulate)
	{
		//获取cloth vertices的指针
		
		glm::vec4* pos; 
		size_t num_bytes;

		cudaError_t cudaStatus = cudaGraphicsMapResources(1, &p_simulate->cuda_vbo_resource, 0);
		cudaStatus = cudaGraphicsResourceGetMappedPointer((void **)&pos, &num_bytes, p_simulate->cuda_vbo_resource);
		p_simulate->VerletCUDA(pos,p_simulate->cloth_vertex_size, p_simulate->cloth_index_size,DEFAULT_DAMPING, mass, timeStep, p_simulate->normalStride,
			finished, p_simulate->gpu_nodes, p_simulate->gpu_indexes, p_simulate->gpu_points);
		cudaStatus = cudaGraphicsUnmapResources(1, &p_simulate->cuda_vbo_resource, 0);
		p_simulate->swap_buffer();
	}
	for (auto object : pscene->objs)
		pscene->RenderBuffer(*object);

}
void Scene::onRender()
{
	getFPS();

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glLoadIdentity();
	glTranslatef(0, dy, 0);
	glTranslatef(0, 0, dist);
	glRotatef(rX, 1, 0, 0);
	glRotatef(rY, 0, 1, 0);

	glGetFloatv(GL_MODELVIEW_MATRIX, modelview);
	glGetFloatv(GL_PROJECTION_MATRIX, projection);
	viewDir.x = (float)-modelview[2];
	viewDir.y = (float)-modelview[6];
	viewDir.z = (float)-modelview[10];
	Right = glm::cross(viewDir, Up);

	DrawGrid();
	RenderGPU_CUDA();

	glutSwapBuffers();
}
void Scene::OnReshape(int nw, int nh)
{
	glViewport(0, 0, nw, nh);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(30, (GLfloat)nw / (GLfloat)nh, 0.1f, 100.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glutPostRedisplay();
}
void Scene::OnIdle()
{
	glutPostRedisplay();
}
void Scene::OnMouseMove(int x, int y)
{
	if (selected_index == -1) {
		if (state == 0)
			dist *= (1 + (y - oldY) / 60.0f);
		else
		{
			rY += (x - oldX) / 5.0f;
			rX += (y - oldY) / 5.0f;
		}
	}
	else {
		float delta = 1500 / abs(dist);
		float valX = (x - oldX) / delta;
		float valY = (oldY - y) / delta;
		if (abs(valX)>abs(valY))
			glutSetCursor(GLUT_CURSOR_LEFT_RIGHT);
		else
			glutSetCursor(GLUT_CURSOR_UP_DOWN);


		glm::vec4* ptr = (glm::vec4*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY);
		glm::vec4 oldVal = ptr[selected_index];
		glUnmapBuffer(GL_ARRAY_BUFFER); // unmap it after use

		glm::vec4 newVal;
		newVal.w = 1;
		// if the pointer is valid(mapped), update VBO
		if (ptr) {
			// modify buffer data				
			oldVal.x += Right[0] * valX;

			float newValue = oldVal.y + Up[1] * valY;
			if (newValue>0)
				oldVal.y = newValue;
			oldVal.z += Right[2] * valX + Up[2] * valY;
			newVal = oldVal;
		}

	}
	oldX = x;
	oldY = y;

	glutPostRedisplay();
}
void Scene::OnMouseDown(int button, int s, int x, int y)
{
	if (s == GLUT_DOWN)
	{
		oldX = x;
		oldY = y;
		int window_y = (height - y);
		float norm_y = float(window_y) / float(height / 2.0);
		int window_x = x;
		float norm_x = float(window_x) / float(width / 2.0);

		float winZ = 0;
		glReadPixels(x, height - y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &winZ);
		if (winZ == 1)
			winZ = 0;
		double objX = 0, objY = 0, objZ = 0;
		GLdouble MV1[16], P1[16];
		gluUnProject(window_x, window_y, winZ, MV1, P1, viewport, &objX, &objY, &objZ);
		glm::vec3 pt(objX, objY, objZ);
		int i = 0;

	}

	if (button == GLUT_MIDDLE_BUTTON)
		state = 0;
	else
		state = 1;

	if (s == GLUT_UP) {
		selected_index = -1;
		glutSetCursor(GLUT_CURSOR_INHERIT);
	}
}
void Scene::OnKey(unsigned char key, int, int)
{
	switch (key)
	{
	case 'w':dy -= 0.1; break;
	case 'W':dy -= 0.1; break;
	case 'S':dy += 0.1; break;
	case 's':dy += 0.1; break;
	default:
		break;
	}

	glutPostRedisplay();
}
void Scene::OnShutdown()
{
}