#include <core/maths.h>
#include <core/shader.h>
#include <core/platform.h>

#include "fem.h"
#include "mesher.h"

#include <iostream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <vector>
#include <stdint.h>


// TODO:
//
// 1. Implement a 2D triangle based FEM based simulation with explicit integration using Green's non-linear strain measure
// 2. If stability is a problem then experiment with linear strain measures and implicit integration
// 3. 

using namespace std;
using namespace fem;

int gWidth = 800;
int gHeight = 600;

float gViewLeft = -2.0f;
float gViewBottom = -1.0f;
float gViewWidth = 5.0f;
float gViewAspect = gHeight/float(gWidth);

int gSubsteps = 1;

Vec2 gMousePos;
int gMouseIndex=-1;
float gMouseStrength = 400.0f;

bool gPause=false;
bool gStep=false;

Scene* gScene;
SceneParams gSceneParams;

vector<Particle> gParticles;
vector<Triangle> gTriangles;
vector<Vec3>	 gPlanes;

void Init()

{
	if (gScene)
		DestroyScene(gScene);

	gParticles.resize(0);
	gTriangles.resize(0);
	gPlanes.resize(0);

	/* Single Tri */

	if (0)
	{
		gSceneParams.mGravity = Vec2(0.0f);

		gParticles.push_back(Particle(Vec2(-1.0f, 0.0f), 0.0f));
		gParticles.push_back(Particle(Vec2( 1.0f, 0.0f), 0.0f));
		gParticles.push_back(Particle(Vec2( 0.0f, sqrtf(3.0f)), 1.0f));
		gParticles.push_back(Particle(Vec2( 2.0f, sqrtf(3.0f)), 1.0f));

		gTriangles.push_back(Triangle(0, 1, 2));
		gTriangles.push_back(Triangle(1, 3, 2));
	}

	/* Cantilever Beam */

	if (0)
	{
		gSubsteps = 20;

		gSceneParams.mDrag = 1.0f;
		gSceneParams.mLameLambda = 10000.0f;
		gSceneParams.mLameMu = 10000.0f;
		gSceneParams.mDamping = 200.0f;
		gSceneParams.mDrag = 0.0f;
		gSceneParams.mToughness = 8000.0f;

		const float kDim = 0.1;

		for (int i=0; i < 10; ++i)
		{
			gParticles.push_back(Particle(Vec2(kDim, i*kDim), 1.0f));
			gParticles.push_back(Particle(Vec2(0.0f, i*kDim), 1.0f));
			
			if (i)
			{
				// add quad
				int start = (i-1)*2;

				gTriangles.push_back(Triangle(start+0, start+2, start+1));
				gTriangles.push_back(Triangle(start+1, start+2, start+3));
			}
		}

		if (1)
		{
			gParticles[0].invMass = 0.0f;
			gParticles[1].invMass = 0.0f;
			gParticles[2].invMass = 0.0f;
			gParticles[3].invMass = 0.0f;
		}

		gPlanes.push_back(Vec3(0.0f, 1.0, 0.5f));
	}	

	/* Donut */

	if (1)
	{	
		gSubsteps = 20;

		gSceneParams.mDrag = 1.0f;
		gSceneParams.mLameLambda = 1000.0f;
		gSceneParams.mLameMu = 1000.0f;
		gSceneParams.mDamping = 80.0f;
		gSceneParams.mDrag = 0.0f;
		gSceneParams.mFriction = 0.95f;
		gSceneParams.mToughness = 2000.0f;

		vector<Vec2> torusPoints;
		vector<uint32_t> torusIndices;

		CreateTorus(torusPoints, torusIndices, 0.2f, 0.5f, 12);
	
		for (size_t i=0; i < torusPoints.size(); ++i)
			gParticles.push_back(Particle(torusPoints[i], 1.0f));
		
		for (size_t i=0; i < torusIndices.size(); i+=3)
			gTriangles.push_back(Triangle(torusIndices[i+0], torusIndices[i+1], torusIndices[i+2]));	

		gPlanes.push_back(Vec3(0.0f, 1.0, 0.5f));
		gPlanes.push_back(Vec3(1.0f, 0.0, 1.2f));
	}

	gScene = CreateScene(&gParticles[0], gParticles.size(), &gTriangles[0], gTriangles.size());
}

int FindClosestParticle(Vec2 p)
{
	float minDistSq = FLT_MAX;
	int minIndex = -1;

	for (size_t i=0; i < gParticles.size(); ++i)
	{
		float d = LengthSq(p-gParticles[i].p);
		
		if (d < minDistSq)
		{
			minDistSq = d;
			minIndex = i;
		}		
	}
	return minIndex;
}

void Modify(float dt)
{
	gParticles.resize(NumParticles(gScene));
	gTriangles.resize(NumTriangles(gScene));

	if (gParticles.empty() || gTriangles.empty())
		return;

	// read out particles
	GetParticles(gScene, &gParticles[0]);
	GetTriangles(gScene, &gTriangles[0]);

	if (gMouseIndex != -1)
	{
		gParticles[gMouseIndex].f += gMouseStrength*(gMousePos-gParticles[gMouseIndex].p);
		//gParticles[gMouseIndex].p = gMousePos;

		SetParticles(gScene, &gParticles[0]);
	}
}


void Update()
{
	float dt = 1.0f/60.0f;

	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(gViewLeft, gViewLeft+gViewWidth, gViewBottom, gViewBottom+gViewWidth*gViewAspect);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// update scene 
	SetParams(gScene, gSceneParams);
	SetPlanes(gScene, &gPlanes[0], gPlanes.size());

	double elapsedTime = 0.0f;

	if (!gPause || gStep)
	{
		dt /= gSubsteps;

		double startTime = GetSeconds();

		for (int i=0; i < gSubsteps; ++i)
		{
			Modify(dt);
			Update(gScene, dt);
		}

		elapsedTime = GetSeconds()-startTime;
	
		gStep = false;
	}

	// planes
	glBegin(GL_LINES);
	glColor3f(0.0f, 0.0f, 1.0f);

	for (size_t i=0; i < gPlanes.size(); ++i)
	{
		Vec2 n = Vec2(gPlanes[i]);
		Vec2 c = gPlanes[i].z * -n;

		glVertex2fv(c + PerpCCW(n)*100.0f);
		glVertex2fv(c - PerpCCW(n)*100.0f);
	}

	glEnd();

	// tris
	glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
	glBegin(GL_TRIANGLES);
	glColor3f(1.0f, 1.0f, 1.0f);
	
	for (size_t i=0; i < gTriangles.size(); ++i)
	{
		Triangle& t = gTriangles[i];

		Vec2 a = gParticles[t.i].p;
		Vec2 b = gParticles[t.j].p;
		Vec2 c = gParticles[t.k].p;
	
		// debug draw	
		glVertex2fv(a);
		glVertex2fv(b);
		glVertex2fv(c);	
	}

	glEnd();

	// particles
	glPointSize(4.0f);
	glBegin(GL_POINTS);
	glColor3f(0.0f, 1.0f, 0.0f);

	for (size_t i=0; i < gParticles.size(); ++i)
		glVertex2fv(gParticles[i].p);

	glEnd();

	// forces
	glBegin(GL_LINES);
	glColor3f(1.0f, 0.0f, 0.0f);

	for (size_t i=0; i < gParticles.size(); ++i)
	{
		const float s = 0.001f;

		glVertex2fv(gParticles[i].p);
		glVertex2fv(gParticles[i].p + gParticles[i].f*s);
	}
	glEnd();
		

	// mouse spring
	if (gMouseIndex != -1)
	{
		glBegin(GL_LINES);
		glColor3f(1.0f, 0.0f, 0.0f);
		glVertex2fv(gMousePos);
		glVertex2fv(gParticles[gMouseIndex].p);
		glEnd();
	}
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, gWidth, gHeight, 0);

	int x = 10;
	int y = 15;
	
	glColor3f(1.0f, 1.0f, 1.0f);
	DrawString(x, y, "Time: %.2fms", float(elapsedTime)*1000.0f); y += 13;

	DrawString(x, y, "Lambda: %.2f", gSceneParams.mLameLambda); y += 13;
	DrawString(x, y, "Mu: %.2f", gSceneParams.mLameMu); y += 13;
	DrawString(x, y, "Toughness: %.2f", gSceneParams.mToughness); y += 13;

	glutSwapBuffers();
}

Vec2 RasterToScene(int x, int y)
{
	float vx = gViewLeft + gViewWidth*x/float(gWidth);
	float vy = gViewBottom + gViewWidth*gViewAspect*(1.0f-y/float(gHeight));
	
	return Vec2(vx, vy);
}

int lastx = 0;
int lasty = 0;

void GLUTMouseFunc(int b, int state, int x, int y)
{	
	switch (state)
	{
		case GLUT_UP:
		{
			lastx = x;
			lasty = y;

			gMouseIndex = -1;
			
			break;
		}	
		case GLUT_DOWN:
		{
			lastx = x;
			lasty = y;

			gMousePos = RasterToScene(x, y);
			gMouseIndex = FindClosestParticle(gMousePos);
		}
	};
}

void GLUTKeyboardDown(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 'w':
		{
			break;
		}
		case 's':
		{
			break;
		}
		case 'a':
		{
			break;
		}
		case 'd':
		{
			break;
		}
		case 't':
		{
			gSceneParams.mToughness += 100.0f;
			break;
		}
		case 'g':
		{
			gSceneParams.mToughness -= 100.0f;
			break;
		}
		case 'u':
		{
			gSceneParams.mLameMu += 100.0f;
			break;
		}
		case 'j':
		{
			gSceneParams.mLameMu -= 100.0f;
			break;
		}
		case 'i':
		{
			gSceneParams.mLameLambda += 100.0f;
			break;
		}
		case 'k':
		{
			gSceneParams.mLameLambda -= 100.0f;
			break;
		}
		case 'p':
		{
			gPause = !gPause;
			break;
		}
		case 'r':
		{
			Init();
			break;
		}
		case 32:
		{
			gPause = true;
			gStep = true;
			break;
		}
		case 27:
		case 'q':
		{
			exit(0);
			break;
		}
	};
}

void GLUTKeyboardUp(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 'w':
		{
			break;
		}
		case 's':
		{
			break;
		}
		case 'a':
		{
			break;
		}
		case 'd':
		{
			break;
		}
	};
}

void GLUTMotionFunc(int x, int y)
{	
//	int dx = x-lastx;
//	int dy = y-lasty;
	
	lastx = x;
	lasty = y;
	
	gMousePos = RasterToScene(x, y);
}

int main(int argc, char* argv[])
{
	// init gl
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	
	glutInitWindowSize(gWidth, gHeight);
	glutCreateWindow("FEM");
	glutPositionWindow(200, 100);

#if _WIN32
	glewInit();
#endif

	Init();

	glutIdleFunc(Update);	
	glutDisplayFunc(Update);
	glutMouseFunc(GLUTMouseFunc);
	glutMotionFunc(GLUTMotionFunc);
	glutKeyboardFunc(GLUTKeyboardDown);
	glutKeyboardUpFunc(GLUTKeyboardUp);
/*	

	glutReshapeFunc(GLUTReshape);
	glutDisplayFunc(GLUTUpdate);
   
	glutSpecialFunc(GLUTArrowKeys);
	glutSpecialUpFunc(GLUTArrowKeysUp);

*/
#if __APPLE__
	int swap_interval = 1;
	CGLContextObj cgl_context = CGLGetCurrentContext();
	CGLSetParameter(cgl_context, kCGLCPSwapInterval, &swap_interval);
#endif

	glutMainLoop();
	return 0;
}





