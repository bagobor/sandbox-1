#include <core/maths.h>
#include <core/shader.h>
#include <core/platform.h>
#include <core/shader.h>
#include <core/tga.h>

#define STRINGIFY(A) #A

#include <iostream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <vector>
#include <stdint.h>

#include "tag.h"
#include "shaders.h"

#if __APPLE__
#include <mach-o/dyld.h>
#endif

using namespace std;

int gScreenWidth = 1280;
int gScreenHeight = 720;

// default file
const char* gFile = "Drawing001.bvh";

Vec3 gCamPos(0.0f);//, 150.0f, -357.0f);
Vec3 gCamVel(0.0f);
Vec3 gCamAngle(kPi, 0.0f, 0.0f);
float gTagWidth = 5.0f;
float gTagHeight = 8.0f;
Point3 gTagCenter;
Point3 gTagLower;
Point3 gTagUpper;
float gTagSmoothing = 0.8f;

GLuint gMainShader;
GLuint gDebugShader;
GLuint gShadowFrameBuffer;
GLuint gShadowTexture;

enum EventType
{
	eStartPaint,
	eStopPaint
};

struct Control
{
	double time;
	EventType event;
};

vector<Control> gControlTrack; 
bool gControlRecord = true;
double gControlStartTime;

struct Frame
{
	Vec3 pos;
	Vec3 rot;
};

bool LoadBvh(const char* filename, std::vector<Frame>& frames, Point3& center, Point3& lower, Point3& upper)
{
	FILE* f = fopen(filename, "r");
	
	lower = Point3(FLT_MAX);
	upper = Point3(-FLT_MAX);

	if (f)
	{
		while (!feof(f))
		{
			Frame frame;
			int n = fscanf(f, "%f %f %f %f %f %f", 
					&frame.pos.x,	
					&frame.pos.y,	
					&frame.pos.z,	
					&frame.rot.z,	
					&frame.rot.x,	
					&frame.rot.y);	
	
			if (n == EOF)
				break;	

			if (n != 6)
			{
				char buf[1024];
				fgets(buf, 1024, f);
			}
			else
			{
				frames.push_back(frame);

				center += frame.pos;
				lower = Min(lower, Point3(frame.pos));
				upper = Max(upper, Point3(frame.pos));
			}
		}

		fclose(f);
	
		center /= frames.size();

		return true;
	}
	else
		return false;
}

vector<Frame> gFrames;
float gFrameRate = 0.01f;
float gFrameTime = 0.0f;
size_t gMaxFrame = 4000;
bool gPause = false;
bool gWireframe = false;
bool gShowNormals = true;
bool gFreeCam = true;

size_t CurrentFrame()
{
	return gFrameTime / gFrameRate;
}

const char* GetPath(const char* file)
{
#if __APPLE__

	static char path[PATH_MAX];
	uint32_t size = sizeof(path);
	_NSGetExecutablePath(path, &size);

	char* lastSlash = strrchr(path, '/');
	strcpy(lastSlash+1, file);
	return path;
#else
	return file;
#endif
}

void Init()
{
	const char* path = GetPath(gFile);

	LoadBvh(path, gFrames, gTagCenter, gTagLower, gTagUpper);

	printf("Finished loading %s.\n", path); 

	gCamPos = Vec3(gTagCenter - Vec3(0.0f, 0.0f, gTagUpper.z-gTagLower.z));
}

void DrawBasis(const Matrix44& m)
{
	glPushMatrix();
	glMultMatrixf(m);
	
	glBegin(GL_LINES);

	glColor3f(1.0f, 0.0f, 0.0f);
	glVertex3fv(Vec3(0.0f));
	glVertex3fv(Vec3(10.0f, 0.0f, 0.0f));

	glColor3f(0.0f, 1.0f, 0.0f);
	glVertex3fv(Vec3(0.0f));
	glVertex3fv(Vec3(0.0f, 10.0f, 0.0f));

	glColor3f(0.0f, 0.0f, 1.0f);
	glVertex3fv(Vec3(0.0f));
	glVertex3fv(Vec3(0.0f, 0.0f, 10.0f));

	glEnd();	
	glPopMatrix();
}

void DrawPlane(Vec3 n, float d)
{
 	const float size = 1000.0f;

	// calculate point on the plane
	Vec3 p = n*d;

	// two vectors that span the plane
	Vec3 a, b;
	BasisFromVector(n, &a, &b);
	
	a *= size;
	b *= size;

	glBegin(GL_QUADS);

	glNormal3fv(n);
	glVertex3fv(p + a + b);
	glVertex3fv(p - a + b);
	glVertex3fv(p - a - b);
	glVertex3fv(p + a - b);

	glEnd();
}

void ShadowCreate(GLuint& texture, GLuint& frameBuffer)
{
	glVerify(glGenFramebuffers(1, &frameBuffer));
	glVerify(glGenTextures(1, &texture));
	glVerify(glBindTexture(GL_TEXTURE_2D, texture));

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
	 
	// This is to allow usage of shadow2DProj function in the shader 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_COMPARE_R_TO_TEXTURE); 
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_LEQUAL); 
	glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY); 

	glVerify(glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, 1024, 1024, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL));

}

void ShadowBegin(GLuint texture, GLuint frameBuffer)
{
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, frameBuffer));
	glVerify(glDrawBuffer(GL_NONE));
	glVerify(glReadBuffer(GL_NONE));
	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture, 0));

	glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT);
	glViewport(0, 0, 1024, 1024);
}

void ShadowEnd()
{
	glVerify(glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, 0, 0));
	glVerify(glBindFramebuffer(GL_FRAMEBUFFER, 0));
	glViewport(0, 0, gScreenWidth, gScreenHeight);
}

void Advance(float dt)
{
	if (!gPause)
		gFrameTime += dt;

	size_t currentFrame = CurrentFrame();

	if (currentFrame >= gFrames.size() || currentFrame > gMaxFrame)
	{
		gFrameTime = 0.0f;
		currentFrame = CurrentFrame();
	}

	// re-create the tag each frame
	Tag tag(gTagSmoothing, gTagWidth, gTagHeight);

	uint32_t control = 0;

	// build tag mesh
	for (size_t i=0; i < currentFrame; ++i)
	{
		double t = i*gFrameRate;
		while (control < gControlTrack.size() && gControlTrack[control].time < t)
		{
			if (gControlTrack[control].event == eStartPaint)
			{
				tag.Start();
			}
			else
				tag.Stop();

			++control;
		}

		// let it run a few frames
	//	if (i == 10)
	//j		tag.Start();
		/*
		Matrix44 m = RotationMatrix(DegToRad(gFrames[i].rot.z), Vec3(0.0f, 0.0f, 1.0f))*
					 RotationMatrix(DegToRad(gFrames[i].rot.x), Vec3(1.0f, 0.0f, 0.0f))*
					 RotationMatrix(DegToRad(gFrames[i].rot.y), Vec3(0.0f, 1.0f, 0.0f));

		m.SetTranslation(Point3(gFrames[i].pos));
		*/
		Matrix44 m = TranslationMatrix(Point3(gFrames[i].pos));

		tag.PushSample(0.0f, m);
	}

	tag.Stop();

	glPolygonMode(GL_FRONT_AND_BACK, gWireframe?GL_LINE:GL_FILL);

	Point3 lightPos = gTagCenter + Vec3(0.0, 250.0, -100.0);
	Point3 lightTarget = gTagCenter;

	Matrix44 lightPerspective = ProjectionMatrix(60.0f, 1.0f, 1.0f, 1000.0f);
   	Matrix44 lightView = LookAtMatrix(lightPos, lightTarget);
	Matrix44 lightTransform = lightPerspective*lightView;
	
	// shadowing pass 
	//
	ShadowBegin(gShadowTexture, gShadowFrameBuffer);
	glMatrixMode(GL_PROJECTION);
	glPushMatrix();
	glLoadMatrixf(lightPerspective);

	glMatrixMode(GL_MODELVIEW);
	glPushMatrix();
	glLoadMatrixf(lightView);

	tag.Draw();

	ShadowEnd();

	glMatrixMode(GL_PROJECTION);
	glPopMatrix();

	glMatrixMode(GL_MODELVIEW);
	glPopMatrix();

	// lighting pass
	//
	glUseProgram(gMainShader);

	GLint uDiffuse = glGetUniformLocation(gMainShader, "shDiffuse");
	glUniform3fv(uDiffuse, 9, reinterpret_cast<float*>(&gShDiffuse[0].x));

	GLint uLightTransform = glGetUniformLocation(gMainShader, "lightTransform");
	glUniformMatrix4fv(uLightTransform, 1, false, lightTransform);

	GLint uLightPos = glGetUniformLocation(gMainShader, "lightPos");
	glUniform3fv(uLightPos, 1, lightPos);
	
	GLint uLightDir = glGetUniformLocation(gMainShader, "lightDir");
	glUniform3fv(uLightDir, 1, Normalize(lightTarget-lightPos));
	
	GLint uColor = glGetUniformLocation(gMainShader, "color");
	glUniform3fv(uColor, 1, Vec3(235.0f/255.0f, 244.0f/255.0f, 223.0f/255.0f));	

	const Vec2 taps[] = 
	{ 
	   	Vec2(-0.326212,-0.40581),Vec2(-0.840144,-0.07358),
		Vec2(-0.695914,0.457137),Vec2(-0.203345,0.620716),
		Vec2(0.96234,-0.194983),Vec2(0.473434,-0.480026),
		Vec2(0.519456,0.767022),Vec2(0.185461,-0.893124),
		Vec2(0.507431,0.064425),Vec2(0.89642,0.412458),
		Vec2(-0.32194,-0.932615),Vec2(-0.791559,-0.59771) 
	};

	GLint uShadowTaps = glGetUniformLocation(gMainShader, "shadowTaps");
	glUniform2fv(uShadowTaps, 12, &taps[0].x);
	
	glEnable(GL_TEXTURE_2D);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, gShadowTexture);

	Point3 lower, upper;
	tag.GetBounds(lower, upper);

	lower = gTagLower;
	upper = gTagUpper;

	const Vec3 margin(0.5f, 0.1f, 0.5f);

	Vec3 edge = upper-lower;
	lower -= edge*margin;
	upper += edge*margin;

	// draw walls 
	DrawPlane(Vec3(0.0f, 1.0f, 0.0f), lower.y);	
	DrawPlane(Vec3(0.0f, 0.0f, -1.0f), -upper.z);
	DrawPlane(Vec3(1.0f, 0.0f, 0.0f), lower.x);	
	DrawPlane(Vec3(-1.0f, 0.0f, 0.0f), -upper.x);
	//DrawPlane(Vec3(0.0f, -1.0f, 0.0f), -gTagUpper.y); 


	// draw the tag, in Mountain Dew green 
	glUniform3fv(uColor, 1, Vec3(1.0f));//Vec3(0.1f, 1.0f, 0.1f));	
	
	if (gShowNormals)
		glUseProgram(gDebugShader);

	tag.Draw();
		
	glDisable(GL_TEXTURE_2D);
	glUseProgram(0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	DrawBasis(tag.basis);
}

void Update()
{
	const float dt = 1.0f/60.0f;
	static float t = 0.0f;
	//t += dt;

	// update camera
	const Vec3 forward(-sinf(gCamAngle.x)*cosf(gCamAngle.y), sinf(gCamAngle.y), -cosf(gCamAngle.x)*cosf(gCamAngle.y));
	const Vec3 right(Normalize(Cross(forward, Vec3(0.0f, 1.0f, 0.0f))));
	
	gCamPos += (forward*gCamVel.z + right*gCamVel.x)*dt;

	glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0f, float(gScreenWidth)/gScreenHeight, 10.0f, 10000.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	
	if (gFreeCam)
	{
		glRotatef(RadToDeg(-gCamAngle.x), 0.0f, 1.0f, 0.0f);
		glRotatef(RadToDeg(-gCamAngle.y), cosf(-gCamAngle.x), 0.0f, sinf(-gCamAngle.x));	
		glTranslatef(-gCamPos.x, -gCamPos.y, -gCamPos.z);
	}
	else
	{
		const float radius = 250.0f;
		const float speed = 0.15f;
		const float alpha = sinf(t*speed)*kPi*0.18f;		

		Vec3 eye = Vec3(gTagCenter) + Vec3(sinf(alpha)*radius, 50.0f, -cosf(alpha)*radius);
		gluLookAt(eye.x, eye.y, eye.z, gTagCenter.x, gTagCenter.y, gTagCenter.z, 0.0f, 1.0f, 0.0f);

		gCamPos = eye;
	}

	double startTime = GetSeconds();

	Advance(dt);
	
	double elapsedTime = GetSeconds()-startTime;

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D(0, gScreenWidth, gScreenHeight, 0);

	int x = 10;
	int y = 15;
	
	char line[1024];

	glColor3f(1.0f, 1.0f, 1.0f);
	sprintf(line, "Draw Time: %.2fms", float(elapsedTime)*1000.0f);
	DrawString(x, y, line); y += 13;

	sprintf(line, "Anim Time: %.2fs", gFrameTime);
	DrawString(x, y, line); y += 13;

	sprintf(line, "Anim Frame: %d", int(CurrentFrame()));
	DrawString(x, y, line); y += 26;

	DrawString(x, y, "1 - New Control Track"); y += 13;

	if (gControlRecord)
	{
		glColor3f(1.0f, 0.0f, 0.0f);
		DrawString(x, y, "recording");
	}

	glutSwapBuffers();

	/* enable to dump frames for video
	
	static int i=0;
	char buffer[255];
	sprintf(buffer, "dump/frame%d.tga", ++i);
		
	TgaImage img;
	img.m_width = gScreenWidth;
	img.m_height = gScreenHeight;
	img.m_data = new uint32_t[gScreenWidth*gScreenHeight];
		
	glReadPixels(0, 0, gScreenWidth, gScreenHeight, GL_RGBA, GL_UNSIGNED_BYTE, img.m_data);
		
	TgaSave(buffer, img);
		
	delete[] img.m_data;
	*/
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
		
			if (gControlRecord)
			{	
				Control c;
				c.time = gFrameTime;//GetSeconds()-gControlStartTime;
			    c.event = eStopPaint;

				gControlTrack.push_back(c);
			}
			break;
		}	
		case GLUT_DOWN:
		{
			lastx = x;
			lasty = y;

			if (gControlRecord)
			{	
				Control c;
				c.time = gFrameTime;//GetSeconds()-gControlStartTime;
			    c.event = eStartPaint;	

				gControlTrack.push_back(c);
			}
			break;
		}
	};
}

void GLUTKeyboardDown(unsigned char key, int x, int y)
{
	const float kSpeed = 120.0f;

	switch (key)
	{
		case 'w':
		{
			gCamVel.z = kSpeed;
			break;
		}
		case 's':
		{
			gCamVel.z = -kSpeed;
			break;
		}
		case 'a':
		{
			gCamVel.x = -kSpeed;
			break;
		}
		case 'd':
		{
			gCamVel.x = kSpeed;
			break;
		}
		case 'u':
		{
			gFrameRate += 0.001f;
			break;
		}
		case 'j':
		{
			gFrameRate -= 0.001f;
			gFrameRate = max(0.001f, gFrameRate);
			break;
		}
		case 'r':
		{
			gFrameTime = 0.0f;
			break;
		}
		case ' ':
		{
			gPause = !gPause;
			break;
		}
		case 'b':
		{
			gWireframe = !gWireframe;
			break;
		}
		case 'n':
		{
			gShowNormals = !gShowNormals;
			break;
		}
		case 'i':
		{
			gTagWidth += 0.1f;
			break;
		}
		case 'k':
		{
			gTagWidth -= 0.1f;
			break;
		}
		case 'o':
		{
			gTagHeight += 0.1f;
			break;
		}
		case 'l':
		{
			gTagHeight -= 0.1f;
			break;
		}
		case 'g':
		{
			gFreeCam = !gFreeCam;
			break;
		}
		case '1':
		{
			if (!gControlRecord)
			{
				gFrameTime = 0.0f;

				gControlTrack.clear();
				gControlRecord = true;
				gControlStartTime = GetSeconds();
			}
			else
			{
				const char* path = GetPath("control.txt");

				FILE* f = fopen(path, "w");
		
				if (f)
				{	
					for (uint32_t i=0; i < gControlTrack.size(); ++i)
					{
						fprintf(f, "%f %i\n", gControlTrack[i].time, gControlTrack[i].event);	
					}

					fclose(f);
				}

				gControlRecord = false;
			}
			break;
		}

		case 'y':
		{
			Tag tag(gTagSmoothing, gTagWidth, gTagHeight);

			// build tag mesh
			for (size_t i=0; i < CurrentFrame(); ++i)
			{
				Matrix44 m = TranslationMatrix(Point3(gFrames[i].pos));
				tag.PushSample(0.0f, m); 
			}

			// export
			tag.ExportToObj("dew.obj");	
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
			gCamVel.z = 0.0f;
			break;
		}
		case 's':
		{
			gCamVel.z = 0.0f;
			break;
		}
		case 'a':
		{
			gCamVel.x = 0.0f;
			break;
		}
		case 'd':
		{
			gCamVel.x = 0.0f;
			break;
		}
	};
}

void GLUTMotionFunc(int x, int y)
{	
	int dx = x-lastx;
	int dy = y-lasty;
	
	lastx = x;
	lasty = y;

	const float kSensitivity = DegToRad(0.1f);

	gCamAngle.x -= dx*kSensitivity;
	gCamAngle.y += dy*kSensitivity;
}

void GLUTReshape(int x, int y)
{
	gScreenWidth = x;
	gScreenHeight = y;
}

int main(int argc, char* argv[])
{
	if (argc < 2)
		printf("BVH file not specified, defaulting to %s.\n", gFile);
	else
		gFile = argv[1];

	// init gl
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_MULTISAMPLE);
	
	glutInitWindowSize(gScreenWidth, gScreenHeight);
	glutCreateWindow("Dew");
	glutPositionWindow(200, 100);

#if _WIN32
	glewInit();	
#endif

	gMainShader = CompileProgram(vertexShader, fragmentShaderMain);
	if (gMainShader == 0)
		return 0;

	gDebugShader = CompileProgram(vertexShader, fragmentShaderDebug);
	if (gDebugShader == 0)
		return 0;

	ShadowCreate(gShadowTexture, gShadowFrameBuffer);

	Init();

	glutIdleFunc(Update);	
	glutDisplayFunc(Update);
	glutMouseFunc(GLUTMouseFunc);
	glutMotionFunc(GLUTMotionFunc);
	glutKeyboardFunc(GLUTKeyboardDown);
	glutKeyboardUpFunc(GLUTKeyboardUp);
	glutReshapeFunc(GLUTReshape);

#if __APPLE__
	// enable v-sync
	int swap_interval = 1;
	CGLContextObj cgl_context = CGLGetCurrentContext();
	CGLSetParameter(cgl_context, kCGLCPSwapInterval, &swap_interval);
#endif

	glutMainLoop();
	return 0;
}





