#include <core/types.h>
#include <core/maths.h>
#include <core/platform.h>
#include <core/hashgrid.h>
#include <core/shader.h>

#include "solve.h"

#include <iostream>

using namespace std;

const uint32_t kWidth = 800;
const uint32_t kHeight = 600;
const float kWorldSize = 2.0f;
const float kZoom = kWorldSize*2.5f;

int kNumParticles = 0;
const int kNumIterations = 10;
const float kDt = 1.0f/60.0f;
const float kRadius = 0.05f;

GrainSystem* g_grains;
GrainParams g_params;

vector<Vec2> g_positions;
vector<Vec2> g_velocities;
vector<float> g_radii;

// mouse
static int lastx;
static int lasty;

Vec2 ScreenToScene(int x, int y)
{
	float aspect = float(kWidth)/kHeight;

	float left = -kZoom*aspect;
	float right =  kZoom*aspect;
	float bottom = -0.5;
	float top = 2*kZoom-0.5f;
	
	float tx = x / float(kWidth);
	float ty = 1.0f - (y / float(kHeight));

	return Vec2(left + tx*(right-left), bottom + ty*(top-bottom));
}

void Init()
{	
	// push back world particle
//	g_positions.push_back(Vec2());
//	g_velocities.push_back(Vec2());
//	g_radii.push_back(0.0f);
	
	g_positions.resize(0);
	g_velocities.resize(0);
	g_radii.resize(0);

	if (true)
	{
		for (int x=0; x < 64; ++x)
		{
			float s = -5.0f;

			for (int i=0; i < 32; ++i)
			{
				s += 2.1f*kRadius;// + Randf(0.0f, 0.05f)*kRadius;

				g_positions.push_back(Vec2(s, kRadius +  kRadius + x*2.0f*kRadius));
				g_velocities.push_back(Vec2());
				g_radii.push_back(kRadius);// + kRadius*Randf(-0.1f, 0.0f));
			}
		}
	}
	else
	{
		g_positions.push_back(Vec2(0.0f, 1.0f));
		g_velocities.push_back(Vec2());
		g_radii.push_back(kRadius);// + kRadius*Randf(-0.1f, 0.0f));
		
			
		g_positions.push_back(Vec2(0.0f, 1.0f + 2.5f*kRadius));
		g_velocities.push_back(Vec2());
		g_radii.push_back(kRadius);// + kRadius*Randf(-0.1f, 0.0f));
	}	

	kNumParticles = g_positions.size();

	g_grains = grainCreateSystem(kNumParticles);
		
	g_params.mGravity = Vec2(0.0f, -9.8f);
	g_params.mDamp = 0.0f;//powf(0.1f, float(kNumIterations));
	g_params.mBaumgarte = 0.5f;
	g_params.mFriction = 0.8f;
	g_params.mRestitution = 0.1f;
	g_params.mOverlap = kRadius*0.1f;
	g_params.mPlanes[0] = Vec3(0.0f, 1.0f, 0.0f);
	g_params.mPlanes[1] = Vec3(1.0f, 0.0f, -5.0f);
	g_params.mPlanes[2] = Vec3(-1.0f, 0.0f, -5.0f);
	g_params.mNumPlanes = 3;

	//g_radii[0] = 2.0f;

	grainSetParams(g_grains, &g_params);
	grainSetPositions(g_grains, (float*)&g_positions[0], kNumParticles);
	grainSetVelocities(g_grains, (float*)&g_velocities[0], kNumParticles);
	grainSetRadii(g_grains, &g_radii[0]);
}

void Shutdown()
{
	grainDestroySystem(g_grains);
}

void Reset()
{
	Shutdown();
	Init();
}

void DrawCircle(const Vec2& p, float r, const Colour& c )
{
	glBegin(GL_TRIANGLE_FAN);
	glColor3f(c.r, c.g, c.b);
	glVertex2fv(p);
	
	const int kSegments = 40;
	for (int i=0; i < kSegments+1; ++i)
	{
		float theta = k2Pi*float(i)/kSegments;
		
		float y = p.y + r*Cos(theta);
		float x = p.x + r*Sin(theta);
		
		glVertex2f(x, y);		
	}
	
	glEnd();
	
}

void DrawString(int x, int y, const char* s)
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, kWidth, kHeight, 0);
	
	glRasterPos2d(x, y);
	while (*s)
	{
		glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *s);
		++s;
	}
}

bool g_step = false;

void GLUTUpdate()
{
	//---------------------------

	glDisable(GL_CULL_FACE);
	glClear(GL_COLOR_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	
	glPointSize(5.0f);

	float aspect = float(kWidth)/kHeight;
	float viewWidth = kZoom*aspect;
	
	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(OrthographicMatrix(-viewWidth, viewWidth, -0.5, 2*kZoom-0.5f, 0.0f, 1.0f));


	// Step
	GrainTimers timers;

	grainSetParams(g_grains, &g_params);
	grainUpdateSystem(g_grains, kDt, kNumIterations, &timers);


	for (int i=0; i < g_params.mNumPlanes; ++i)
	{	
		Vec2 p = g_params.mPlanes[i].z * Vec2(g_params.mPlanes[i].x, g_params.mPlanes[i].y);
		Vec2 d = Vec2(-g_params.mPlanes[i].y, g_params.mPlanes[i].x);
		
		glBegin(GL_LINES);
		glColor3f(1.0f, 1.0f, 1.0f);
		glVertex2fv(p - d*1000.0);
		glVertex2fv(p + d*1000.0);
		glEnd();		
	}
		
	// read-back data	
	grainGetPositions(g_grains, (float*)&g_positions[0]);
	
	glColor3f(0.7f, 0.7f, 0.8f);

	double drawStart = GetSeconds();

	glPointSize(kRadius*kWidth/viewWidth);
	glEnable(GL_BLEND);

	Colour colors[] = { Colour(0.5f, 0.5f, 0.8f),
					Colour(0.8f, 0.5f, 0.5f),
					Colour(0.5f, 0.8f, 0.5f) };

	
	glBegin(GL_POINTS);

	for (int i=0; i < kNumParticles; ++i)
	{
		glColor3fv(colors[i%3]);
		glVertex2fv(g_positions[i]);
//		DrawCircle(g_positions[i], kRadius, colors[i%3]);
	}

	glEnd();
	glDisable(GL_BLEND);

	double drawEnd = GetSeconds();
	
	Vec2 mouse = ScreenToScene(lastx, lasty);
	DrawCircle(mouse, 0.2f, Colour(1.0f, 0.0f, 0.0f));
		
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, kWidth, kHeight, 0);

	int x = 10;
	int y = 15;
	
	glColor3f(1.0f, 1.0f, 1.0f);
	DrawString(x, y, "Draw time: %.2fms", (drawEnd-drawStart)*1000.0f); y += 13;
	DrawString(x, y, "Create Cell Indices: %.2fms", timers.mCreateCellIndices); y += 13;
	DrawString(x, y, "Sort Cell Indices: %.2fms", timers.mSortCellIndices); y += 13;
	DrawString(x, y, "Create Grid: %.2fms", timers.mCreateGrid); y += 13;
	DrawString(x, y, "Collide: %.2fms", timers.mCollide); y += 13;
	DrawString(x, y, "Integrate: %.2fms", timers.mIntegrate); y += 13;
	DrawString(x, y, "Reorder: %.2fms", timers.mReorder); y += 13;

	glutSwapBuffers();
	
}

void GLUTReshape(int width, int height)
{
}

void GLUTArrowKeys(int key, int x, int y)
{
}

void GLUTArrowKeysUp(int key, int x, int y)
{
}

void GLUTKeyboardDown(unsigned char key, int x, int y)
{
 	switch (key)
	{
		case 'e':
		{
			break;
		}
		case 'r':
		{
			Reset();
			break;
		}
		case 't':
		{
			g_params.mNumPlanes--;
			break;
		}
		case 's':
		{
			g_step = true;
			break;
		}
		case 'q':
		case 27:
			exit(0);
			break;
	};
}

void GLUTKeyboardUp(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 'e':
		{
			break;
		}
	}
}

void GLUTMouseFunc(int b, int state, int x, int y)
{	
	switch (state)
	{
		case GLUT_UP:
		{
			lastx = x;
			lasty = y;
			
			break;
		}
		case GLUT_DOWN:
		{
			lastx = x;
			lasty = y;
		}
	}
}

void GLUTMotionFunc(int x, int y)
{	
    int dx = x-lastx;
    int dy = y-lasty;
	
	lastx = x;
	lasty = y;

	
	g_positions[0] = ScreenToScene(x, y);
	g_velocities[0] = Vec2(dx*5.0f, 5.0f*(1.0f-dy));

	grainSetPositions(g_grains, (float*)&g_positions[0], 1);
	grainSetVelocities(g_grains, (float*)&g_velocities[0], 1);
	
}

int solveCuda(float* a, float* b, float* c, int n);

int main(int argc, char* argv[])
{	
//	float a[8] = {0, 1, 2, 3, 4, 5, 6, 7};
//	float b[8] = {1, 1, 1, 1, 1, 1, 1, 1};
//	float c[8] = {0};
//	
//	cout << solveCuda(a, b, c, 8);
//	
//	for (int i=0; i < 8; ++i)
//		cout << c[i] << endl;
	
	RandInit();
	Init();
	
    // init gl
    glutInit(&argc, argv);

#ifdef WIN32
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
#endif
	
    glutInitWindowSize(kWidth, kHeight);
    glutCreateWindow("Granular");
    glutPositionWindow(200, 100);
		
    glutMouseFunc(GLUTMouseFunc);
    glutReshapeFunc(GLUTReshape);
    glutDisplayFunc(GLUTUpdate);
    glutKeyboardFunc(GLUTKeyboardDown);
    glutKeyboardUpFunc(GLUTKeyboardUp);
    glutIdleFunc(GLUTUpdate);	
    glutSpecialFunc(GLUTArrowKeys);
    glutSpecialUpFunc(GLUTArrowKeysUp);
    glutMotionFunc(GLUTMotionFunc);

#ifndef WIN32
	int swap_interval = 1;
	CGLContextObj cgl_context = CGLGetCurrentContext();
	CGLSetParameter(cgl_context, kCGLCPSwapInterval, &swap_interval);
#endif

    glutMainLoop();

}

