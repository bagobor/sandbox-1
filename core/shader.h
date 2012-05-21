#pragma once

#if _WIN32

#include <external/glew/include/gl/glew.h>
#include <external/glut/glut.h>

#elif __APPLE__

#include <opengl/opengl.h>
#include <glut/glut.h>

#elif PLATFORM_IOS

#if OGL1
#import <OpenGLES/EAGL.h>
#import <OpenGLES/ES1/gl.h>
#import <OpenGLES/ES1/glext.h>
#else
#import <OpenGLES/ES2/gl.h>
#import <OpenGLES/ES2/glext.h>
#endif

#endif

#include "core/vec4.h"

#define glVerify(x) {x; glAssert(#x, __LINE__, __FILE__);}
void glAssert(const char* msg, long line, const char* file);

GLuint CompileProgramFromFile(const char *vertexPath, const char *fragmentPath);
GLuint CompileProgram(const char *vsource, const char *fsource);

void DrawPlane(const Vec4& p);
void DrawString(int x, int y, const char* s, ...);

