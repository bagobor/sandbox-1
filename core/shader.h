#pragma once

#if _WIN32

#include <External/glew/include/gl/glew.h>
#include <External/freeglut-2.6.0/include/gl/glut.h>

#elif __APPLE__

#include <OpenGL/OpenGL.h>
#include <GLUT/glut.h>

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

#include "Core/Vec4.h"

#define glVerify(x) {x; glAssert(#x, __LINE__, __FILE__);}
void glAssert(const char* msg, long line, const char* file);

GLuint CompileProgram(const char *vsource, const char *fsource);

void DrawPlane(const Vec4& p);
//void DrawString(int x, int y, const char* s);
void DrawString(int x, int y, const char* s, ...);
