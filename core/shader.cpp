#include "Shader.h"

#include "Types.h"
#include "Maths.h"

#include <stdarg.h>
#include <stdio.h>

#define WITH_GLEW

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

void glAssert(const tchar* msg, long line, const char* file)
{
	struct glError 
	{
		GLenum code;
		const tchar* name;
	};

	static const glError errors[] = {	{GL_NO_ERROR, _TS("No Error")},
										{GL_INVALID_ENUM, _TS("Invalid Enum")},
										{GL_INVALID_VALUE, _TS("Invalid Value")},
										{GL_INVALID_OPERATION, _TS("Invalid Operation")}
#if OGL1
										,{GL_STACK_OVERFLOW, _TS("Stack Overflow")},
										{GL_STACK_UNDERFLOW, _TS("Stack Underflow")},
										{GL_OUT_OF_MEMORY, _TS("Out Of Memory")}
#endif
									};

	GLenum e = glGetError();

	if (e == GL_NO_ERROR)
	{
		return;
	}
	else
	{
		const char* errorName = "Unknown error";

		// find error message
		for (uint32 i=0; i < sizeof(errors)/sizeof(glError); i++)
		{
			if (errors[i].code == e)
			{
				errorName = errors[i].name;
			}
		}

		printf("OpenGL: %s - error %s in %s at line %d\n", msg, errorName, file, int(line));
		assert(0);
	}
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
	else
	{
		printf("Created shader program: %d\n", program);
	}

	return program;
}

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

void DrawStringA(int x, int y, const char* s)
{
	glRasterPos2d(x, y);
	while (*s)
	{
		glutBitmapCharacter(GLUT_BITMAP_8_BY_13, *s);
		++s;
	}
}

void DrawString(int x, int y, const char* s, ...)
{
	char buf[2048];

	va_list args;

	va_start(args, s);
	vsprintf(buf, s, args);
	va_end(args);

	DrawStringA(x ,y, buf);
}

