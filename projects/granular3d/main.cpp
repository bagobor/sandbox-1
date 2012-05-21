#include <core/types.h>
#include <core/maths.h>
#include <core/platform.h>
#include <core/hashgrid.h>
#include <core/maths.h>
#include <core/shader.h>

#include <iostream>

using namespace std;

#include "cuda.h"
#include "cuda_runtime.h"  

#include "solve.h"

const uint32_t kWidth = 800;
const uint32_t kHeight = 600;

int kNumParticles = 0;
const int kNumIterations = 5;
const float kRadius = 0.1f;

GrainSystem* g_grains;
GrainParams g_params;

vector<Vec3> g_positions;
vector<Vec3> g_velocities;
vector<float> g_radii;

Vec3 g_camPos(0.0f, 10.0f,  20.0f);
Vec3 g_camAngle(0.0f, -kPi/6.0f, 0.0f);

// mouse
static int lastx;
static int lasty;

// render funcs
void DrawPoints(float* positions, int n, float radius, float screenWidth, float screenAspect);
void DrawPlane(const Vec4& p);

// size of initial grid of particles
const int kParticleHeight = 128;
const int kParticleWidth = 32;
const int kParticleLength = 32;

void Init()
{	
	g_positions.resize(0);
	g_velocities.resize(0);
	g_radii.resize(0);

	float y = kRadius;//2.0f + kRadius;

	for (int i=0; i < kParticleHeight; ++i)
	{	
		for (int z=0; z < kParticleLength; ++z)
		{
			for (int x=0; x < kParticleWidth; ++x)
			{
				float s = 0.0f;

				g_positions.push_back(Vec3(x*2.4f*kRadius + Randf(0.0f, 0.2f*kRadius), y, z*2.2f*kRadius + Randf(0.0f, 0.2f*kRadius)));				
				g_velocities.push_back(Vec3());
				g_radii.push_back(kRadius);
			}
		}

		y += 2.f*kRadius;
	}

	kNumParticles = g_positions.size();

	g_grains = grainCreateSystem(kNumParticles);
		
	float r2 = 1.0f / sqrtf(2.0f);

	g_params.mGravity = make_float3(0.0f, -9.8f, 0.0f);
	g_params.mDamp = powf(0.3f, float(kNumIterations));
	g_params.mBaumgarte = 0.2f;
	g_params.mFriction = 0.5f;
	g_params.mRestitution = 0.1f;
	g_params.mOverlap = kRadius*0.05f;
	g_params.mPlanes[0] = make_float4(0.0f, 1.0f, 0.0f, 0.0f);
	g_params.mPlanes[1] = make_float4(1.0f, 0.0f, 0.0f, 5.0f);
	g_params.mPlanes[2] = make_float4(-r2, r2, 0.0f, 6.0f);
	g_params.mPlanes[3] = make_float4(0.0f, 0.0f, -1.0f, 10.0f);
	g_params.mPlanes[4] = make_float4(0.0f, 0.0f, 1.0f, 5.0f);
	g_params.mNumPlanes = 5;
	
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


bool g_step = false;

void GLUTUpdate()
{
	GrainTimers timers;

	grainSetParams(g_grains, &g_params);
	grainUpdateSystem(g_grains, 1.0f/60.0f, kNumIterations, &timers);

	//---------------------------

	glEnable(GL_CULL_FACE);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	
	glPointSize(5.0f);

	float aspect = float(kWidth)/kHeight;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, aspect, 0.1f, 1000.0f);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glRotatef(RadToDeg(-g_camAngle.x), 0.0f, 1.0f, 0.0f);
	glRotatef(RadToDeg(-g_camAngle.y), cosf(-g_camAngle.x), 0.0f, sinf(-g_camAngle.x));	
	glTranslatef(-g_camPos.x, -g_camPos.y, -g_camPos.z);

	// planes
	for (int i=0; i < g_params.mNumPlanes; ++i)
	{	
		DrawPlane(Vec4((float*)&(g_params.mPlanes[i])));
	}
	
	// read-back data	
	grainGetPositions(g_grains, (float*)&g_positions[0]);
	grainGetRadii(g_grains, (float*)&g_radii[0]);
	
	glColor3f(0.7f, 0.7f, 0.8f);

	float hue = 0.2f;
	float kGoldenRatioConjugate = 0.61803398874989f;

	double drawStart = GetSeconds();

	DrawPoints(&g_positions[0].x, kNumParticles, kRadius, float(kWidth), float(kWidth)/kHeight);

	glDisable(GL_LIGHTING);
	glDisable(GL_BLEND);

	double drawEnd = GetSeconds();
	
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
	DrawString(x, y, "Particles: %d", kParticleHeight*kParticleWidth*kParticleLength, kNumIterations); y += 13;
	DrawString(x, y, "Iterations: %d", kNumIterations); y += 26;

	DrawString(x, y, "t - Remove plane"); y += 13;
	DrawString(x, y, "u - Move plane"); y += 13;
	DrawString(x, y, "r - Reset"); y += 13;
		
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
	const float kSpeed = 0.5f;

	// update camera
	const Vec3 forward(-sinf(g_camAngle.x)*cosf(g_camAngle.y), sinf(g_camAngle.y), -cosf(g_camAngle.x)*cosf(g_camAngle.y));
	const Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));
	
	Vec3 delta;

 	switch (key)
	{
		case 'w':
		{
			g_camPos += kSpeed*forward;
			break;
		}
		case 's':
		{
			g_camPos -= kSpeed*forward;
			break;
		}
		case 'a':
		{
			g_camPos -= kSpeed*right;
			break;
		}
		case 'd':
		{
			g_camPos += kSpeed*right;
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
		case 'u':
		{
			g_params.mPlanes[2].w -= 0.1f;
			break;
		}

		case 27:
			exit(0);
			break;
	};

	g_camPos += delta;
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

	const float kSensitivity = DegToRad(0.1f);

	g_camAngle.x -= dx*kSensitivity;
	g_camAngle.y -= dy*kSensitivity;
}

int solveCuda(float* a, float* b, float* c, int n);


int main(int argc, char* argv[])
{	
	RandInit();
	Init();
	
    // init gl
    glutInit(&argc, argv);

#ifdef WIN32
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	
    glutInitWindowSize(kWidth, kHeight);
    glutCreateWindow("Granular");
    glutPositionWindow(200, 200);
	
	glewInit();

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

