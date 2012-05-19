// FogVolumes.cpp : 
//

#include "Graphics/ParticleContainer.h"
#include "Graphics/ParticleEmitter.h"
#include "Graphics/RenderGL/GLUtil.h"
#include "Core/Perlin.h"

#include "External/glew/include/GL/glew.h"
#include "External/freeglut-2.6.0/include/GL/freeglut.h"

#include <vector>

using namespace std;

uint32 g_width = 1280;
uint32 g_height = 720;

GLuint g_depthBuffer;
GLuint g_noiseTexture;

// one for each light type
GLuint g_staticPrograms[2];
GLuint g_particlePrograms[2];

GLuint g_particleTexture;

ParticleContainer* g_particleContainer;
ParticleEmitter* g_particleEmitter;

enum LightType 
{
	kLightSpot=0,
	kLightPoint
};

struct Light
{	
	LightType m_type;
	
	Matrix44 m_lightToWorld;
	Vector4 m_lightColour;	
};

Light g_lights[2];

int g_selectedLightIndex = 0;
int g_activeLights = 2;
float g_scatteringCoefficient = 0.3f;
bool g_showHelp =  true;
bool g_showParticles = false;

Light& GetSelectedLight() { return g_lights[g_selectedLightIndex]; }

GLuint CreateNoiseTexture(int width, int height, int depth)
{
	float *data = new float [width*height*depth*4];
	float *ptr = data;
	
	int period = 7;
	float frequency = 7 / float(width-1);

	for(int x=0; x < width; x++)
	{
		for (int y=0; y < height; ++y)
		{
			for (int z=0; z < depth; ++z)
			{
				float r = Perlin3DPeriodic(x * frequency, y * frequency, z * frequency, period, period, period, 1, 1.0f); 
				float g = Perlin3DPeriodic(100.0f + x * frequency, y * frequency, z * frequency, period, period, period, 1, 1.0f); 
				float b = Perlin3DPeriodic(x * frequency, 100.0f + y * frequency, z * frequency, period, period, period, 1, 1.0f); 
				float a = Perlin3DPeriodic(x * frequency, y * frequency, 100.0f + z * frequency, period, period, period, 1, 1.0f); 
				
				*ptr++ = r;
				*ptr++ = g;
				*ptr++ = b;
				*ptr++ = a;
			}
		}
	}

	GLuint texid;
	glGenTextures(1, &texid);
	GLenum target = GL_TEXTURE_3D;
	glBindTexture(target, texid);

	glTexParameteri(target, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(target, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(target, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(target, GL_TEXTURE_WRAP_R, GL_REPEAT);

	GlVerify(glPixelStorei(GL_UNPACK_ALIGNMENT, 1));
	GlVerify(glTexImage3D(target, 0, GL_RGBA16F_ARB, width, height, depth, 0, GL_RGBA, GL_FLOAT, data));
	delete [] data;

	return texid;
}

void Init()
{ 
	g_staticPrograms[kLightSpot] = GlslCreateProgramFromFiles("Data/SimpleVertex.glsl", "Data/SpotLight.glsl");
	g_staticPrograms[kLightPoint]= GlslCreateProgramFromFiles("Data/SimpleVertex.glsl", "Data/PointLight.glsl");

	g_particlePrograms[kLightSpot] = GlslCreateProgramFromFiles("Data/SimpleVertex.glsl", "Data/SpotLightParticle.glsl");
	g_particlePrograms[kLightPoint] = GlslCreateProgramFromFiles("Data/SimpleVertex.glsl", "Data/PointLightParticle.glsl");

	// create some default lights
	g_lights[0].m_lightToWorld = TransformMatrix(Rotation(), Point3(5.0f, 10.0f, 0.0f));
	g_lights[0].m_lightColour = Vector4(0.3f, 0.3f, 0.8f, 1.0f)*4.0f;
	g_lights[0].m_type = kLightSpot ;

	g_lights[1].m_lightToWorld = TransformMatrix(Rotation(), Point3(-5.0f, 10.0f, 0.0f));
	g_lights[1].m_lightColour = Vector4(0.8f, 0.3f, 0.3f, 1.0f)*4.0f;
	g_lights[1].m_type = kLightPoint;

	// create a noise texture
	//g_noiseTexture = CreateNoiseTexture(128, 128, 128); 

	// particle texture
	g_particleTexture = GlCreateTextureFromFile("data/smoke.tga");

	RandInit();

	// create a simple particle system
	g_particleContainer = new ParticleContainer(500);
	g_particleContainer->m_additive = true;
	g_particleContainer->m_acceleration = Vec4(0.0f, 0.0f, 0.0, 0.0f);
	g_particleContainer->m_collisionEnabled = false;
	g_particleContainer->m_scale.LoadFromString("1, 0, 1");
	g_particleContainer->m_stretchAmount = 0.1f;
	g_particleContainer->m_sort = false;
	g_particleContainer->m_linearDrag = 0.0f;
	g_particleContainer->m_angularDrag = 0.0f;
	g_particleContainer->m_curlNoiseStrength = 0.0f;

	g_particleEmitter = new ParticleEmitter();
	g_particleEmitter->SetWorldTransform(TransformMatrix(Rotation(), Point3(0.0f, 7.0f, 10.0f)));
	g_particleEmitter->SetContainer(g_particleContainer);
	g_particleEmitter->m_spawnSize = 2.5f;
	g_particleEmitter->m_spawnSizeRand = 1.0f;
	g_particleEmitter->m_spawnColourRand = Vec4(0.0f, 0.0f, 0.0f, 0.0f);
	g_particleEmitter->m_spawnLifetime = 200.0f;
	g_particleEmitter->m_spawnSpeed = 0.0f;
	g_particleEmitter->m_spawnSpeedRand = 0.0f;
	g_particleEmitter->m_spawnPositionRand = 10.0f;
	g_particleEmitter->m_spawnPositionRand = Vec4(10.0f, 4.0f, 10.f, 0.0f);
	g_particleEmitter->m_spawnRotationSpeedRand = 100.0f;	
	g_particleEmitter->m_emitRate = 1000;
	g_particleEmitter->m_emitInstant = 1000;
	
}

void Shutdown()
{

}

void RenderParticles(ParticleContainer& container, GLuint program)
{
	if (container.m_numActiveParticles == 0)
		return;

	// setup blend modes
	glEnable(GL_BLEND);

	if (container.m_additive)
		glBlendFunc(GL_ONE, GL_ONE);
	else
		glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

	GlVerify(glEnable(GL_TEXTURE_2D));
	GlVerify(glActiveTexture((GLenum)GL_TEXTURE0));	
	GlVerify(glBindTexture(GL_TEXTURE_2D, g_particleTexture));

	//GLuint param = glGetUniformLocation(program, "g_noiseTexture");
	//GlVerify(glUniform1i(param, (GLint)0)); // texunit 0

	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glDisable(GL_CULL_FACE);
	glDisable(GL_LIGHTING);
	glDepthMask(false);
	glDisable(GL_DEPTH_TEST);
	// generate particle geometry
	vector<ParticleVertex> vb(container.m_numActiveParticles*4);

		// get model view matrix and invert
	Matrix44 view;
	glGetFloatv(GL_MODELVIEW_MATRIX, (float*)&view);

	view = AffineInverse(view);

	container.BuildVertexBuffer(&vb[0], view);

	// set vertex buffers
	glEnableClientState(GL_VERTEX_ARRAY);	
	glEnableClientState(GL_COLOR_ARRAY);
	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);

	glVertexPointer(3, GL_FLOAT, sizeof(ParticleVertex), &vb[0].position);
	glColorPointer(4, GL_FLOAT, sizeof(ParticleVertex), &vb[0].colour);
	glTexCoordPointer(2, GL_FLOAT, sizeof(ParticleVertex), &vb[0].uv);

	// draw
	GlVerify(glDrawArrays(GL_QUADS, 0, (GLsizei)container.m_numActiveParticles*4));

	// reset state
	glDisableClientState(GL_VERTEX_ARRAY);	
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_TEXTURE_COORD_ARRAY);

	// reset state
	glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glDepthMask(true);
	glEnable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDisable(GL_TEXTURE_2D);

}

void RenderStaticGeometry()
{	
	glDisable(GL_TEXTURE_2D);

	// draw scene geometry
	Matrix44 xform = TransformMatrix(Rotation(0.0f, 0.0f, 0.0f), Point3(5.0f, 5.0f, 0.0f));
	GlDrawSphere(xform, 2.0f);	

	glBegin(GL_QUADS);
	
	glNormal3f(0.0f, 1.0f, 0.0f);
	glVertex4f(-200.0f, -2.0f, 200.0f, 1.0f);
	glVertex4f(200.0f, -2.0f, 200.0f, 1.0f);
	glVertex4f(200.0f, -2.0f, -200.0f, 1.0f);
	glVertex4f(-200.0f, -2.0f, -200.0f, 1.0f);

	glNormal3f(0.0f, 0.0f, 1.0f);
	glVertex4f(-200.0f, -200.0f, -2.0f, 1.0f);
	glVertex4f(200.0f, -200.0f, -2.0f, 1.0f);
	glVertex4f(200.0f, 200.0f, -2.0f, 1.0f);
	glVertex4f(-200.0f, 200.0f, -2.0f, 1.0f);
	
	glEnd();
}


void SetLightParams(Light& l, GLuint program)
{
	// disable depth test for full screen quad
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_BLEND);
	glDepthMask(true);
	glBlendFunc(GL_ONE, GL_ONE);
	glEnable(GL_CULL_FACE);
	glDisable(GL_LIGHTING);
	glDisable(GL_ALPHA_TEST);
	glDepthFunc(GL_LEQUAL);

	glUseProgram(program);
	
	GLuint param = glGetUniformLocation(program, "g_lightToWorld");
	glUniformMatrix4fv(param, 1, false, l.m_lightToWorld);

	param = glGetUniformLocation(program, "g_worldToLight");
	glUniformMatrix4fv(param, 1, false, AffineInverse(l.m_lightToWorld));

	param = glGetUniformLocation(program, "g_lightCol");
	glUniform4fv(param, 1, l.m_lightColour);

	param = glGetUniformLocation(program, "g_scatteringCoefficient");
	glUniform1f(param, g_scatteringCoefficient);

	/*
	param = glGetUniformLocation(program, "g_noiseTexture");
	if (param != -1)
	{
		GlVerify(glActiveTexture((GLenum)GL_TEXTURE1));	
		GlVerify(glEnable(GL_TEXTURE_3D));
		GlVerify(glBindTexture(GL_TEXTURE_3D, g_noiseTexture));
	}
	*/
	
}


void RenderHelp()
{	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0.0f, g_width, g_height, 0.0f);

	glUseProgram(0);

	int x = 50;
	int y = 50;
	int dy = 12;

	glColor4f(1.0f, 1.0f, 1.0f, 1.0f);

	glRasterPos2i(x, y);
	glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*)"space - switch light"); y += dy;

	glRasterPos2i(x, y);
	glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*)"left drag - translate"); y += dy;

	glRasterPos2i(x, y);
	glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*)"right drag - rotate"); y += dy;

	glRasterPos2i(x, y);
	glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*)"t - change light type"); y += dy;

	glRasterPos2i(x, y);
	glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*)"p - toggle particles"); y += dy;

	char buffer[1024];

	sprintf(buffer, "u,j - scattering coefficient (%.3f)", g_scatteringCoefficient);
	glRasterPos2i(x, y);
	glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*)buffer); y += dy;

	sprintf(buffer, "i,k - light intensity (%.2f, %.2f, %.2f)", GetSelectedLight().m_lightColour.x, GetSelectedLight().m_lightColour.y, GetSelectedLight().m_lightColour.z); 
	glRasterPos2i(x, y);
	glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*)buffer); y += dy;

	glRasterPos2i(x, y);
	glutBitmapString(GLUT_BITMAP_8_BY_13, (const unsigned char*)"h - toggle help"); y += dy;

}

void Render()
{
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0f, 20.0f, 25.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, float(g_width) / g_height, 0.1f, 1000.0f);
	
	// z-pre pass
	glColorMask(false, false, false, false);
	
	RenderStaticGeometry();

	glColorMask(true, true, true, true);

	for (int i=0; i < g_activeLights; ++i)
	{
		Light& l = g_lights[i];

		
		// select shader program and render
		GLuint staticProgram = g_staticPrograms[l.m_type];
		SetLightParams(l, staticProgram);

		RenderStaticGeometry();

		// same for particles
		if (g_showParticles)
		{
			GLuint particleProgram = g_particlePrograms[l.m_type];
			SetLightParams(l, particleProgram);

			RenderParticles(*g_particleContainer, particleProgram);
		}
	}

	// debug
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_BLEND);
	
	if (g_showHelp)
	{
		GlDrawBasis(GetSelectedLight().m_lightToWorld);	

		RenderHelp();
	}
}



//--------------------------------------------------------------------------
// Glut callbacks

void GlutUpdate()
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	static double s_lastUpdate = GetSeconds();
	
	double curTime = GetSeconds();
	float dt = float(curTime - s_lastUpdate);
	s_lastUpdate = curTime;

	g_particleEmitter->Update(dt);
	g_particleContainer->Update(dt);
	
	Render();

	// flip
	glutSwapBuffers();
}

void GlutReshape(int width, int height)
{
	g_width = width;
	g_height = height;

	glViewport(0, 0, width, height);
}

void GlutArrowKeys(int key, int x, int y)
{
}

void GlutArrowKeysUp(int key, int x, int y)
{
}

void GlutKeyboardDown(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 'u':
		{
			g_scatteringCoefficient += 0.001f;
			break;
		}
	case 'j':
		{
			g_scatteringCoefficient = max(0.0f, g_scatteringCoefficient-0.001f);
			break;
		}
	case 't':
		{
			// swap light type
			GetSelectedLight().m_type = (GetSelectedLight().m_type == kLightPoint) ? kLightSpot : kLightPoint;
			break;
		}
	case ' ':
		{
			g_selectedLightIndex = (g_selectedLightIndex+1) % g_activeLights;
			break;
		}
	case 'i':
		{
			GetSelectedLight().m_lightColour *= 1.1f;
			break;
		}
	case 'k':
		{
			GetSelectedLight().m_lightColour *= 0.9f;
			break;
		}
	case 'p':
		{
			g_showParticles = !g_showParticles;
			break;
		}
	case 'h':
		{
			g_showHelp = !g_showHelp;
			break;
		}
	case 27:
		exit(0);
		break;
	};

}

void GlutKeyboardUp(unsigned char key, int x, int y)
{
}

static int lastx;
static int lasty;
static int button;

void GlutMouseFunc(int b, int state, int x, int y)
{
	switch (state)
	{
	case GLUT_UP:
		{
			lastx = x;
			lasty = y;			
		}
	case GLUT_DOWN:
		{
			lastx = x;
			lasty = y;
			button = b;
		}
	}
}

void GlutMotionFunc(int x, int y)
{
	int dx = x - lastx;
	int dy = y - lasty;

	lastx = x;
	lasty = y;
	
	Light& selectedLight = g_lights[g_selectedLightIndex];
	
	if (button == GLUT_LEFT_BUTTON)
	{
		Point3 t = selectedLight.m_lightToWorld.GetTranslation();
		
		t.x += dx * 0.01f;
		t.z += dy * 0.01f;

		selectedLight.m_lightToWorld.SetTranslation(t);
	}
	else
	{
		selectedLight.m_lightToWorld *= RotationMatrix(dx * 0.01f, Vec3(1.0f, 0.0f, 0.0f));
		selectedLight.m_lightToWorld *= RotationMatrix(dy * 0.01f, Vec3(0.0f, 0.0f, 1.0f));
	}
}

int main(int argc, char* argv[])
{
	// init glc
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH );

	glutInitWindowSize(g_width, g_height);
	glutCreateWindow("Fog Volumes");
	glutPositionWindow(200, 200);

	glewInit();
	if (glewIsSupported("GL_VERSION_2_0"))
	{
		Log::Info << "Ready for OpenGL 2.0" << endl;
	}
	else 
	{
		Log::Warn << "OpenGL 2.0 not supported" << endl;
		return 1;
	}

	float one = 1.0f;

	Init();

	glutMouseFunc(GlutMouseFunc);
	glutReshapeFunc(GlutReshape);
	glutDisplayFunc(GlutUpdate);
	glutKeyboardFunc(GlutKeyboardDown);
	glutKeyboardUpFunc(GlutKeyboardUp);
	glutIdleFunc(GlutUpdate);	
	glutSpecialFunc(GlutArrowKeys);
	glutSpecialUpFunc(GlutArrowKeysUp);
	glutMotionFunc(GlutMotionFunc);

	glutMainLoop();

	Shutdown();

	return 0;	
}

