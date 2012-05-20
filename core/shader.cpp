#include "Shader.h"

#include "Types.h"
#include "Maths.h"
#include "Platform.h"
#include "Tga.h"
#include "Png.h"

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

void PreProcessShader(const char* filename, std::string& source)
{
	// load source
	FILE* f = fopen(filename, "r");

	if (!f)
	{
		printf("Could not open shader file for reading: %s\n", filename);
		return;
	}

	// add lines one at a time handling include files recursively
	while (!feof(f))
	{
		char buf[1024];

		if (fgets(buf, 1024, f) != NULL)
		{	
			// test for #include
			if (strncmp(buf, "#include", 8) == 0)
			{	
				const char* begin = strchr(buf, '\"');
				const char* end = strrchr(buf, '\"');

				if (begin && end && (begin != end))
				{
					// lookup file relative to current file
					PreProcessShader((StripFilename(filename) + std::string(begin+1, end)).c_str(), source);
				}
			}
			else
			{
				// add line to output
				source += buf;
			}
		}
	}

	fclose(f);
}

GLuint CompileProgramFromFile(const char *vertexPath, const char *fragmentPath)
{
	std::string vsource;
	PreProcessShader(vertexPath, vsource);

	std::string fsource;
	PreProcessShader(fragmentPath, fsource);

	return CompileProgram(vsource.c_str(), fsource.c_str());
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

GLuint CreateTextureFromFile(const char* path, float* w, float* h, bool clamp)
{
	uint32* imageData = NULL;
	uint16 imageWidth = 0;
	uint16 imageHeight = 0;

	const char* extension = strrchr(path, '.');
	if (stricmp(extension, ".png") == 0)
	{
		PngImage image;
		if (PngLoad(path, image))
		{
			imageWidth = image.m_width;
			imageHeight = image.m_height;
			imageData = image.m_data;
		}
	}
	else if (stricmp(extension, ".tga") == 0)
	{
		
		FILE* aTGAFile = fopen(path, "rb");
		if (aTGAFile == NULL)
		{
			printf("Texture: could not open %s for reading.\n", path);
			return NULL;
		}

		char aHeaderIDLen;
		fread(&aHeaderIDLen, sizeof(byte), 1, aTGAFile);

		char aColorMapType;
		fread(&aColorMapType, sizeof(byte), 1, aTGAFile);
	
		char anImageType;
		fread(&anImageType, sizeof(byte), 1, aTGAFile);

		short aFirstEntryIdx;
		fread(&aFirstEntryIdx, sizeof(u16), 1, aTGAFile);

		short aColorMapLen;
		fread(&aColorMapLen, sizeof(u16), 1, aTGAFile);

		char aColorMapEntrySize;
		fread(&aColorMapEntrySize, sizeof(byte), 1, aTGAFile);	

		short anXOrigin;
		fread(&anXOrigin, sizeof(u16), 1, aTGAFile);

		short aYOrigin;
		fread(&aYOrigin, sizeof(u16), 1, aTGAFile);

		short anImageWidth;
		fread(&anImageWidth, sizeof(u16), 1, aTGAFile);	

		short anImageHeight;
		fread(&anImageHeight, sizeof(u16), 1, aTGAFile);	

		char aBitCount = 32;
		fread(&aBitCount, sizeof(byte), 1, aTGAFile);	

		char anImageDescriptor;// = 8 | (1<<5);
		fread((char*)&anImageDescriptor, sizeof(byte), 1, aTGAFile);


		// total is the number of bytes we'll have to read
		int total = anImageWidth * anImageHeight * 4;

		assert(total);

		// allocate memory for image pixels
		byte* surfaceData = new byte[total];

		// check to make sure we have the memory required
		if (surfaceData  == NULL) 
			return false;

		// finally load the image pixels
		if (fread(reinterpret_cast<char*>(surfaceData), total, 1, aTGAFile) != 1)
		{
			printf("Texture: file not fully read, may be corrupt (%s).\n", path);
		}

		// if bit 5 of the descriptor is set then the image is flipped vertically so we fix it up
		if (anImageDescriptor & (1 << 3))
		{
			// swap all the rows
			int rowSize = anImageWidth*4;	

			byte* buf = new byte[anImageWidth*4];
			byte* start = surfaceData;
			byte* end = &surfaceData[rowSize*(anImageHeight-1)];
		
			while (start < end)
			{
				memcpy(buf, end, rowSize);
				memcpy(end, start, rowSize);
				memcpy(start, buf, rowSize);

				start += rowSize;
				end -= rowSize;
			}

			delete[] buf;
		}

		fclose(aTGAFile);

		imageWidth = anImageWidth;
		imageHeight = anImageHeight;
		imageData = (uint32*)surfaceData;
	}

	
	// pre-multiply
	byte* pixel = (byte*)imageData;
	uint32 pixelCount = imageWidth*imageHeight;

	for (uint32 i=0; i < pixelCount; ++i)
	{
		float a = float(pixel[3])/255;
		pixel[0] = byte(pixel[0] * a);
		pixel[1] = byte(pixel[1] * a);
		pixel[2] = byte(pixel[2] * a);

		pixel += 4;
	}
	
	if (w)
		*w = imageWidth;
	if (h)
		*h = imageHeight;
	
	GLuint id;
	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_2D, id);

	if (clamp)
	{
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	}
	else
	{
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	}

	glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_NEAREST);

	glVerify(gluBuild2DMipmaps(GL_TEXTURE_2D, GL_RGBA, imageWidth, imageHeight, GL_RGBA, GL_UNSIGNED_BYTE, imageData));		

	delete[] imageData;
	return id;
}