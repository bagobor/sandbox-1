#include <core/maths.h>
#include <core/shader.h>
#include <core/platform.h>
#include <core/tga.h>

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
float gMouseStrength = 2000.0f;

bool gPause=false;
bool gStep=false;

Scene* gScene;
SceneParams gSceneParams;

vector<Particle> gParticles;
vector<Triangle> gTriangles;
vector<Vec3>	 gPlanes;
vector<Vec2>	 gUVs;

GLuint gTexture;
bool gShowTexture = false;

GLuint CreateTexture(uint32_t width, uint32_t height, void* data)
{
	GLuint id;

	glVerify(glGenTextures(1, &id));
	glBindTexture(GL_TEXTURE_2D, id);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);//_MIPMAP_LINEAR);
	///glTexParameteri(GL_TEXTURE_2D, GL_GENERATE_MIPMAP, GL_TRUE);		
	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data));
	//glGenerateMipmapEXT(GL_TEXTURE_2D);

	return id;
}

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

	/* Random convex */
	
	if (0)
	{	
		gSubsteps = 20;

		gSceneParams.mDrag = 1.0f;
		gSceneParams.mLameLambda = 10000.0f;
		gSceneParams.mLameMu = 10000.0f;
		gSceneParams.mDamping = 80.0f;
		gSceneParams.mDrag = 0.0f;
		gSceneParams.mFriction = 0.95f;
		gSceneParams.mToughness = 2000.0f;

		gPlanes.push_back(Vec3(0.0f, 1.0, 0.5f));

		// generate a random set of points in a unit cube
		RandInit();

		const uint32_t numPoints = 10;
		vector<Vec2> points;

		for (uint32_t i=0; i < numPoints; ++i)
			points.push_back(Vec2(0.0f, 1.0f) + 0.5f*Vec2(Randf(-1.0f, 1.0f), Randf(-1.0f, 1.0f)));

		// triangulate
		vector<uint32_t> tris;
		TriangulateDelaunay(&points[0], points.size(), tris);

		// generate elements
		for (uint32_t i=0; i < points.size(); ++i)
			gParticles.push_back(Particle(points[i], 1.0f));

		for (uint32_t i=0; i < tris.size()/3; ++i)
			gTriangles.push_back(Triangle(tris[i*3], tris[i*3+1], tris[i*3+2]));
	}

	/* Image */
	if (1)
	{
		gSubsteps = 80;

		gSceneParams.mDrag = 1.0f;
		gSceneParams.mLameLambda = 115000.0f;
		gSceneParams.mLameMu = 115000.0f;
		gSceneParams.mDamping = 120.0f;
		gSceneParams.mDrag = 1.0f;
		gSceneParams.mFriction = 0.5f;
		gSceneParams.mToughness = 0.0f;

		gPlanes.push_back(Vec3(0.0f, 1.0, 0.5f));
		gPlanes.push_back(Vec3(1.0f, 0.0, 1.2f));

		TgaImage img;
		TgaLoad("armadillo.tga", img);
		
		gTexture = CreateTexture(img.m_width, img.m_height, img.m_data);

		vector<Vec2> points;

		// distribute points inside the image at evenly spaced intervals
		const float resolution = 0.04f;
		const float scale = 2.0f;
	
		const uint32_t dd = img.m_width*resolution;
		const float aspect = float(img.m_height)/img.m_width;
		const float noise = 0.0f;

		for (uint32_t y=0; y < img.m_height; y+=dd)
		{
			for (uint32_t x=0; x < img.m_width; x+=dd)
			{
				// if value non-zero then add a point
				if (img.m_data[y*img.m_width + x] != 0)
				{
					Vec2 uv(float(x)/img.m_width + Randf(-noise, noise), float(y)/img.m_height + Randf(-noise, noise));
					gUVs.push_back(uv);

					points.push_back(scale*Vec2(uv.x, uv.y*aspect));
				}
			}
		}

		// triangulate
		vector<uint32_t> tris;
		TriangulateDelaunay(&points[0], points.size(), tris);

		// generate elements
		for (uint32_t i=0; i < points.size(); ++i)
			gParticles.push_back(Particle(points[i], 1.0f));

		for (uint32_t i=0; i < tris.size()/3; ++i)
		{
			Triangle t(tris[i*3], tris[i*3+1], tris[i*3+2]);

			Vec2 p = points[t.i];
			Vec2 q = points[t.j];
			Vec2 r = points[t.k];

			// ignore triangles whose center is not inside the shape
			Vec2 c = (p+q+r)/3.0f;
			
			uint32_t x = c.x*img.m_width/scale;
			uint32_t y = c.y*img.m_width/scale;
			
			if (img.m_data[y*img.m_width + x] != 0)
				gTriangles.push_back(t);
		}
	}

	/* Donut */

	if (0)
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

		CreateTorus(torusPoints, torusIndices, 0.2f, 0.5f, 30);
	
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
		Vec2 pq = gMousePos-gParticles[gMouseIndex].p;

		Vec2 damp = -10.0f*Dot(Normalize(pq), gParticles[gMouseIndex].v)*Normalize(pq);
		Vec2 stretch = gMouseStrength*pq;

		gParticles[gMouseIndex].f += stretch + damp; 
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
	if (gTexture && gShowTexture)
	{
		glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D, gTexture);
		
	}
	else
	{
		glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );
		glDisable(GL_TEXTURE_2D);
	}

	glDisable(GL_CULL_FACE);
	glBegin(GL_TRIANGLES);
	glColor3f(1.0f, 1.0f, 1.0f);
	
	for (size_t i=0; i < gTriangles.size(); ++i)
	{
		Triangle& t = gTriangles[i];

		Vec2 a = gParticles[t.i].p;
		Vec2 b = gParticles[t.j].p;
		Vec2 c = gParticles[t.k].p;
	
		if (gTexture && gShowTexture)
			glTexCoord2fv(gUVs[t.i]);
		glVertex2fv(a);
	
		if (gTexture && gShowTexture)
			glTexCoord2fv(gUVs[t.j]);
		glVertex2fv(b);
		
		if (gTexture && gShowTexture)
			glTexCoord2fv(gUVs[t.k]);
		glVertex2fv(c);	
	}

	glEnd();

	if (!gShowTexture)
	{

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
	}

	if (gShowTexture)
		glDisable(GL_TEXTURE_2D);


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
		case 'b':
		{
			gShowTexture = !gShowTexture;
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





