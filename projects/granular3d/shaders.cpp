#include "shaders.h"

#include <Core/Types.h>
#include <Core/Maths.h>

#define WITH_GLEW

#include <Graphics/RenderGL/GLUtil.h>

void GlslPrintShaderLog(GLuint obj)
{
#if !PLATFORM_IOS
	int infologLength = 0;
	int charsWritten  = 0;
	char *infoLog;
	
	glGetShaderiv(obj, GL_INFO_LOG_LENGTH,&infologLength);
	
	if (infologLength > 1)
	{
		infoLog = (char *)malloc(infologLength);
		glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
		printf("%s\n",infoLog);
		free(infoLog);
	}
#endif
}


GLuint CompileProgram(const char *vsource, const char *fsource)
{
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);

	glShaderSource(vertexShader, 1, &vsource, 0);
	glShaderSource(fragmentShader, 1, &fsource, 0);
	
	glCompileShader(vertexShader);
	GlslPrintShaderLog(vertexShader);
	
	glCompileShader(fragmentShader);
	GlslPrintShaderLog(fragmentShader);

	GLuint program = glCreateProgram();

	glAttachShader(program, vertexShader);
	glAttachShader(program, fragmentShader);

	glLinkProgram(program);

	// check if program linked
	GLint success = 0;
	glGetProgramiv(program, GL_LINK_STATUS, &success);

	if (!success) {
		char temp[256];
		glGetProgramInfoLog(program, 256, 0, temp);
		printf("Failed to link program:\n%s\n", temp);
		glDeleteProgram(program);
		program = 0;
	}

	return program;
}

void DrawPoints(float* positions, int n, float radius, float screenWidth, float screenAspect)
{
	static int sprogram = -1;
	if (sprogram == -1)
		sprogram = CompileProgram(vertexShader, fragmentShader);

	if (sprogram)
	{
		glEnable(GL_POINT_SPRITE);
		glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
		glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
		glDepthMask(GL_TRUE);
		glEnable(GL_DEPTH_TEST);		

		glUseProgram(sprogram);
		glUniform1f( glGetUniformLocation(sprogram, "pointScale"), radius*2.0f);
		glUniform1f( glGetUniformLocation(sprogram, "pointRadius"), screenWidth / (2.0f*screenAspect*tanf(kPi/8.0f)));

		glColor3f(1, 1, 1);

		glEnableClientState(GL_VERTEX_ARRAY);			
		glVertexPointer(3, GL_FLOAT, sizeof(float)*3, positions);

		glDrawArrays(GL_POINTS, 0, n);

		glUseProgram(0);
		glDisableClientState(GL_VERTEX_ARRAY);	
		glDisable(GL_POINT_SPRITE_ARB);
	}
}


/*
void 
{
	m_program = compileProgram(vertexShader, spherePixelShader);

#if !defined(__APPLE__) && !defined(MACOSX)
	glClampColorARB(GL_CLAMP_VERTEX_COLOR_ARB, GL_FALSE);
	glClampColorARB(GL_CLAMP_FRAGMENT_COLOR_ARB, GL_FALSE);
#endif
}
*/



void DrawPlane(const Vec4& p)
{
	Vec3 u, v;
	BasisFromVector(Vec3(p.x, p.y, p.z), &u, &v);

	Vec3 c = Vec3(p.x, p.y, p.z)*-p.w;
	
	const float kSize = 100.f;

	glBegin(GL_QUADS);
	glColor3fv(p*0.5f + Vec4(0.5f, 0.5f, 0.5f, 0.5f));
	glVertex3fv(c + u*kSize + v*kSize);
	glVertex3fv(c - u*kSize + v*kSize);
	glVertex3fv(c - u*kSize - v*kSize);
	glVertex3fv(c + u*kSize - v*kSize);
	glEnd();
}

void DrawString(int x, int y, const char* s)
{
	glRasterPos2d(x, y);
	while (*s)
	{
		glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *s);
		++s;
	}
}
