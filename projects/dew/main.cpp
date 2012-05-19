#include <Core/Maths.h>
#include <Core/Shader.h>
#include <Core/Platform.h>
#include <Core/Shader.h>
#include <Core/Tga.h>

#define STRINGIFY(A) #A

#include <iostream>
#include <cmath>
#include <functional>
#include <algorithm>
#include <vector>
#include <stdint.h>

#if __APPLE__
#include <mach-o/dyld.h>
#endif

using namespace std;

int gScreenWidth = 1280;
int gScreenHeight = 720;

// default file
const char* gFile = "Drawing001.bvh";

Vec3 gCamPos(0.0f, 150.0f, -357.0f);
Vec3 gCamVel(0.0f);
Vec3 gCamAngle(kPi, 0.0f, 0.0f);
float gTagWidth = 5.0f;
float gTagHeight = 8.0f;
Point3 gTagCenter;

GLuint gMainShader;
GLuint gDebugShader;
GLuint gShadowFrameBuffer;
GLuint gShadowTexture;

// spherical harmonic coefficients for the 'beach' light probe
Vec3 gShDiffuse[] = 
{
	Vec3(1.51985, 1.53785, 1.56834),
	Vec3(-0.814902, -0.948101, -1.13014),
	Vec3(-0.443242, -0.441047, -0.421306),
	Vec3(1.16161, 1.07284, 0.881858),
	Vec3(-0.36858, -0.37136, -0.332637),
	Vec3(0.178697, 0.200577, 0.219209),
	Vec3(-0.0204381, -0.0136351, -0.00920174),
	Vec3(-0.349078, -0.292836, -0.214752),
	Vec3(0.399496, 0.334641, 0.219389),
};

// vertex shader
const char* vertexShader = STRINGIFY
(
	uniform mat4 lightTransform; 
	
	void main()
	{
		vec3 n = gl_Normal;//normalize(gl_Normal);

		gl_Position = gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xyz, 1.0);
		gl_TexCoord[0] = vec4(n, 0.0);
		gl_TexCoord[1] = vec4(gl_Vertex.xyz, 1.0);
		gl_TexCoord[2] = lightTransform*vec4(gl_Vertex.xyz+n, 1.0);
	}
);

// pixel shader
const char* fragmentShaderMain = STRINGIFY
(
	uniform vec3 shDiffuse[9];
	uniform vec3 color;

	uniform vec3 lightDir;
	uniform vec3 lightPos;

	uniform sampler2DShadow shadowTex;
	uniform vec2 shadowTaps[12];

	// evaluate spherical harmonic function
	vec3 shEval(vec3 dir, vec3 sh[9])
	{
		// evaluate irradiance
		vec3 e = sh[0];

		e += -dir.y*sh[1];
		e +=  dir.z*sh[2];
		e += -dir.x*sh[3];

		e +=  dir.x*dir.y*sh[4];
		e += -dir.y*dir.z*sh[5];
		e += -dir.x*dir.z*sh[7];

		e += (3.0*dir.z*dir.z-1.0)*sh[6];
		e += (dir.x*dir.x - dir.y*dir.y)*sh[8];

		return max(e, vec3(0.0));
	}

	// sample shadow map
	float shadowSample()
	{
		vec3 pos = vec3(gl_TexCoord[2].xyz/gl_TexCoord[2].w);
		vec3 uvw = (pos.xyz*0.5)+vec3(0.5);

		// user clip
		if (uvw.x  < 0.0 || uvw.x > 1.0)
			return 0.0;
		if (uvw.y < 0.0 || uvw.y > 1.0)
			return 0.0;
		
		float s = 0.0;
		float radius = 0.003;

		for (int i=0; i < 12; i++)
		{
			s += shadow2D(shadowTex, vec3(uvw.xy + shadowTaps[i]*radius, uvw.z)).r;
		}

		s /= 12.0;
		return s;
	}

	void main()
	{
		vec3 n = gl_TexCoord[0].xyz;
		vec3 shadePos = gl_TexCoord[1].xyz;
		vec3 eyePos = gl_ModelViewMatrixInverse[3].xyz;
		vec3 eyeDir = normalize(eyePos-shadePos);
	
		vec3 lightCol = vec3(1.0, 1.0, 1.0)*0.8; 

		// SH-ambient	
		float ambientExposure = 0.04;
		vec3 ambient = shEval(n, shDiffuse)*ambientExposure;

		// wrapped spot light 
		float w = 0.1;
		float s = shadowSample();
		vec3 direct = clamp((dot(n, -lightDir) + w) / (1.0 + w), 0.0, 1.0)*lightCol*smoothstep(0.9, 1.0, dot(lightDir, normalize(shadePos-lightPos))); 

		vec3 l = (ambient + s*direct)*color;//*(n*vec3(0.5) + vec3(0.5));//color;

		// convert from linear light space to SRGB
		gl_FragColor = vec4(pow(l, vec3(0.5)), 1.0);
	}
);

// pixel shader
const char* fragmentShaderDebug = STRINGIFY
(
	void main()
	{
		vec3 n = gl_TexCoord[0].xyz;
		gl_FragColor = vec4(n*0.5 + vec3(0.5), 1.0);
	}
);

struct Frame
{
	Vec3 pos;
	Vec3 rot;
};

bool LoadBvh(const char* filename, std::vector<Frame>& frames, Point3& center)
{
	FILE* f = fopen(filename, "r");
	
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
		
			if (n != 6)
				break;

			frames.push_back(frame);

			center += frame.pos;
		}

		fclose(f);
	
		center /= frames.size();

		return true;
	}
	else
		return false;
}

vector<Frame> gFrames;
float gFrameRate = 0.005f;
float gFrameTime = 0.0f;
size_t gMaxFrame = 4000;
bool gPause = false;
bool gWireframe = false;
bool gShowNormals = false;
bool gFreeCam = false;

size_t CurrentFrame()
{
	return gFrameTime / gFrameRate;
}

void Init()
{
#if _APPLE_
	uint32_t size = sizeof(path);
	_NSGetExecutablePath(path, &size);

	char* lastSlash = strrchr(path, '/');
	strcpy(lastSlash+1, gFile);
#else
	const char* path = gFile;
#endif
	
	LoadBvh(path, gFrames, gTagCenter);

	printf("Finished loading %s.\n", path); 
}
struct Vertex
{
	Vertex() {}
	Vertex(Point3 p, Vec3 n) : position(p), normal(n) {}

	Point3 position;
	Vec3 normal;
};

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

void SquareBrush(float t, vector<Vertex>& verts)
{
	float w = gTagWidth;
	float h = gTagHeight;

	const Vertex shape[] = 
	{
		Vertex(Point3(-w,  h, 0.0f), Vec3(-1.0f, 0.0f, 0.0f)),
		Vertex(Point3(-w,  -h, 0.0f), Vec3(-1.0f, 0.0f, 0.0f)),
	
		Vertex(Point3( -w, -h, 0.0f), Vec3( 0.0f, -1.0f, 0.0f)),
		Vertex(Point3( w,  -h, 0.0f), Vec3( 0.0f, -1.0f, 0.0f)),
	
		Vertex(Point3( w, -h, 0.0f), Vec3( 1.0f, 0.0f, 0.0f)),
		Vertex(Point3( w,  h, 0.0f), Vec3( 1.0f, 0.0f, 0.0f)),
		
		Vertex(Point3( w,  h, 0.0f), Vec3( 0.0f, 1.0f, 0.0f)),
		Vertex(Point3( -w, h, 0.0f), Vec3( 0.0f, 1.0f, 0.0f))
	};

	verts.assign(shape, shape+8);

}

struct Tag
{
	Tag() : basis(Matrix44::kIdentity) 
	{
		samples.reserve(4096);
		vertices.reserve(100000);
		indices.reserve(100000);
	}

	void PushSample(float t, Matrix44 m)
	{
		// evaluate brush
		SquareBrush(t, brush);

		size_t startIndex = vertices.size();

		samples.push_back(m.GetTranslation());

		// need at least 4 points to construct valid tangents
		if (samples.size() < 4)
			return;

		// the point we are going to output
		size_t c = samples.size()-3;

		// calculate the tangents for the two samples using central differencing
		Vec3 tc = Normalize(samples[c+1]-samples[c-1]);
		Vec3 td = Normalize(samples[c+2]-samples[c]);
		float a = acosf(Dot(tc, td));

		if (fabsf(a) > 0.001f)
		{
			// use the parallel transport method to move the reference frame along the curve
			Vec3 n = Normalize(Cross(tc, td));

			if (samples.size() == 4)
				basis = TransformFromVector(Normalize(tc));
		
			// 'transport' the basis forward
			basis = RotationMatrix(a, n)*basis;
			
			m = basis;
			m.SetTranslation(samples[c]);
			basis = m;
		}
		
		// transform verts and create faces
		for (size_t i=0; i < brush.size(); ++i)
		{
			// transform position and normal to world space
			Vertex v(m*brush[i].position, m*brush[i].normal);

			vertices.push_back(v);
		}

		if (startIndex != 0)
		{
			size_t b = brush.size();

			for (size_t i=0; i < b; ++i)
			{
				size_t curIndex = startIndex + i;
				size_t nextIndex = startIndex + (i+1)%b; 

				indices.push_back(curIndex);
				indices.push_back(curIndex-b);
				indices.push_back(nextIndex-b);
				
				indices.push_back(nextIndex-b);
				indices.push_back(nextIndex);
				indices.push_back(curIndex);			
			}	
		}
	}

	void Draw()
	{
		if (vertices.empty())
			return;

		// draw the tag
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, sizeof(Vertex), &vertices[0].position);
		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, sizeof(Vertex), &vertices[0].normal);

		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, &indices[0]);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

		glBegin(GL_TRIANGLE_FAN);

		SquareBrush(1.f, brush);

		Point3 center(0.0f);

		// transform verts and create faces
		for (size_t i=0; i < brush.size(); ++i)		
		{
			center += Vec3(brush[i].position);
		}
		
		center /= brush.size();
		
		glNormal3fv(basis.GetCol(2));
		glVertex3fv(basis*center);

		// transform verts and create faces
		for (size_t i=0; i < brush.size(); ++i)
		{
			// transform position and normal to world space
			Vertex v(basis*brush[i].position, basis*brush[i].normal);

			glVertex3fv(v.position);
			
		}

		glEnd();

	}

	void Clear()
	{
		samples.resize(0);
		brush.resize(0);
		vertices.resize(0);
		indices.resize(0);
	}

	Matrix44 basis;

	std::vector<Point3> samples;
	std::vector<Vertex> brush;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;
};

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
	Tag tag;

	// build tag mesh
	for (size_t i=0; i < currentFrame; ++i)
	{
		/*
		Matrix44 m = RotationMatrix(DegToRad(gFrames[i].rot.z), Vec3(0.0f, 0.0f, 1.0f))*
					 RotationMatrix(DegToRad(gFrames[i].rot.x), Vec3(1.0f, 0.0f, 0.0f))*
					 RotationMatrix(DegToRad(gFrames[i].rot.y), Vec3(0.0f, 1.0f, 0.0f));

		m.SetTranslation(Point3(gFrames[i].pos));
		*/
		Matrix44 m = TranslationMatrix(Point3(gFrames[i].pos));

		tag.PushSample(0.0f, m); 
	}

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

	// draw walls 
	DrawPlane(Vec3(0.0f, 1.0f, 0.0f), 0.0f);	
	DrawPlane(Vec3(0.0f, 0.0f, -1.0f), -200.0f);
	DrawPlane(Vec3(1.0f, 0.0f, 0.0f), -200.0f);	
	DrawPlane(Vec3(-1.0f, 0.0f, 0.0f), -200.0f);
	DrawPlane(Vec3(0.0f, -1.0f, 0.0f), -400.0f);


	// draw the tag, in Mountain Dew green 
	glUniform3fv(uColor, 1, Vec3(1.0f));//Vec3(0.1f, 1.0f, 0.1f));	
	
	if (gShowNormals)
		glUseProgram(gDebugShader);

	tag.Draw();
		
	glDisable(GL_TEXTURE_2D);
	glUseProgram(0);
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

float g_dt;

void Update()
{
	const float dt = 1.0f/60.0f;
	static float t = 0.0f;
	t += dt;

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

		Vec3 eye = gTagCenter + Vec3(sinf(alpha)*radius, 50.0f, -cosf(alpha)*radius);
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
	DrawString(x, y, line); y += 13;

	glutSwapBuffers();

	static int i=0;
	char buffer[255];
	sprintf(buffer, "dump/frame%d.tga", ++i);
		
	TgaImage img;
	img.m_width = gScreenWidth;
	img.m_height = gScreenHeight;
	img.m_data = new uint32[gScreenWidth*gScreenHeight];
		
	glReadPixels(0, 0, gScreenWidth, gScreenHeight, GL_RGBA, GL_UNSIGNED_BYTE, img.m_data);
		
	TgaSave(buffer, img);
		
	delete[] img.m_data;

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
			
			break;
		}	
		case GLUT_DOWN:
		{
			lastx = x;
			lasty = y;
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





