#include <iostream>

#include "core/types.h"
#include "core/shader.h"
#include "core/platform.h"

#define STRINGIFY(A) #A

using namespace std;

uint32_t g_screenWidth = 800;
uint32_t g_screenHeight = 600;

GLuint g_pointShader;
GLuint g_solidShader;

//--------------------------------------------------------
// Solid shaders
//
const char *vertexShader = STRINGIFY(

void main()
{
    // calculate window-space point size
	gl_Position = gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xyz, 1.0);
	gl_TexCoord[0] = gl_ModelViewMatrixInverseTranspose*vec4(gl_Normal.xyz, 0.0);   
}
);

const char *fragmentShader = STRINGIFY(

void main()
{
	gl_FragColor = gl_TexCoord[0];
}
);

//--------------------------------------------------------
// Point shaders
//
const char *vertexPointShader = STRINGIFY(

uniform float pointRadius;  // point size in world space
uniform float pointScale;   // scale to calculate size in pixels

void main()
{
	vec4 eyePos = gl_ModelViewMatrix*vec4(gl_Vertex.xyz, 1.0);

    // calculate window-space point size
	//gl_Position = gl_ModelViewProjectionMatrix*vec4(gl_Vertex.xyz, 1.0);
	//gl_PointSize = (pointRadius/gl_Position.w)*pointScale + 4.0;
	
	gl_Position = gl_Vertex;

	gl_TexCoord[0] = gl_MultiTexCoord0;   
}
);

const char* geometryPointShader = 
"#version 120\n"
"#extension GL_EXT_geometry_shader4 : enable\n"
STRINGIFY(
//layout (points) in
//layout (quads, max_vertices=4) out
void main()
{
	vec3 pos = gl_PositionIn[0].xyz;

	vec3 x = vec3(1.0, 0.0, 0.0);
	vec3 y = vec3(0.0, 1.0, 0.0);

	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos-x+y, 1);
	EmitVertex();

	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos-x-y, 1);
	EmitVertex();

	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos+x-y, 1);
	EmitVertex();

	gl_Position = gl_ModelViewProjectionMatrix * vec4(pos+x+y, 1);
	EmitVertex();
}
);

// pixel shader for rendering points as shaded spheres
const char *fragmentPointShader = STRINGIFY(

uniform vec3 invViewport;
uniform vec3 invProjection;
uniform mat4 invQuadric;

bool solveQuadratic(float a, float b, float c, out float minT, out float maxT)
{
	if (a == 0.0 && b == 0.0)
	{
		minT = maxT = 0.0;
		return true;
	}

	float discriminant = b*b - 4.0*a*c;

	if (discriminant < 0.0)
	{
		return false;
	}

	float t = -0.5*(b + sign(b)*sqrt(discriminant));
	minT = t / a;
	maxT = c / t;

	if (minT > maxT)
	{
		float tmp = minT;
		minT = maxT;
		maxT = tmp;
	}

	return true;
}

float sqr(float x) { return x*x; }

void main()
{
	
	vec4 ndcPos = vec4(gl_FragCoord.xy*invViewport.xy*vec2(2.0, 2.0) - vec2(1.0, 1.0), -1.0, 1.0);
	vec4 viewDir = gl_ProjectionMatrixInverse*ndcPos; //vec4(clipPos*invProjection, 0.0); 

	// ray to parameter space
	vec4 dir = invQuadric*gl_ModelViewMatrixInverse*vec4(viewDir.xyz, 0.0);
	vec4 origin = (invQuadric*gl_ModelViewMatrixInverse)[3];

	// set up quadratric equation
	float a = sqr(dir.x) + sqr(dir.y) + sqr(dir.z);// - sqr(dir.w);
	float b = dir.x*origin.x + dir.y*origin.y + dir.z*origin.z - dir.w*origin.w;
	float c = sqr(origin.x) + sqr(origin.y) + sqr(origin.z) - sqr(origin.w); 	

	float minT;
	float maxT;
	
	if (solveQuadratic(a, 2.0*b, c, minT, maxT))
	{
		vec3 hitPos = viewDir.xyz*minT;
		vec3 dx = dFdx(hitPos);
		vec3 dy = dFdy(hitPos);

		gl_FragColor.xyz = normalize(cross(dx, dy));
		gl_FragColor.w = 1.0;

		return;
	}
	//else
	//	discard;	
	
	
	/*
    // calculate normal from texture coordinates
    vec3 normal;
    normal.xy = gl_TexCoord[0].xy*vec2(2.0, -2.0) + vec2(-1.0, 1.0);
    float mag = dot(normal.xy, normal.xy);
    if (mag > 1.0) discard;   // kill pixels outside circle
   	normal.z = sqrt(1.0-mag);
	*/	

	//gl_FragColor = vec4(normal.x, normal.y, normal.z, 1.0);
	gl_FragColor = vec4(0.5, 0.0, 0.0, 1.0);
}


);

// dot product with negative w
float DotInvW(const Vec4& a, const Vec4& b)
{
	return a.x*b.x + a.y*b.y + a.z*b.z - a.w*b.w;
}

void Init()
{
	int maxVerts;
	glGetIntegerv(GL_MAX_GEOMETRY_OUTPUT_VERTICES_EXT, &maxVerts);
	printf("%d\n", maxVerts);
		
	g_pointShader = CompileProgram(vertexPointShader, fragmentPointShader, geometryPointShader);
	g_solidShader = CompileProgram(vertexShader, fragmentShader);
}


void GLUTUpdate()
{	
	glViewport(0, 0, g_screenWidth, g_screenHeight);
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

	const float fov = 45.0f;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(fov, g_screenWidth/float(g_screenHeight), 0.01f, 1000.0f);
	
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	gluLookAt(0.0f, 0.0f, 10.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

	float radius = 1.0f;
	float aspect = float(g_screenWidth)/g_screenHeight;

	Point3 quadricPos = Point3(1.f*radius, 0.0f, 0.0f);
	Vec3 quadricScale = Vec3(0.5f, 1.0f, 1.0f);

	Matrix44 T = TranslationMatrix(quadricPos)*RotationMatrix(0.0f*kPi/4.0f, Vec3(0.0f, 0.0f, 1.0f)); 
	Matrix44 S = ScaleMatrix(quadricScale);

	Matrix44 TInv = AffineInverse(T);
	Matrix44 SInv = ScaleMatrix(Vec3(1.0f/quadricScale.x, 1.0f/quadricScale.y, 1.0f/quadricScale.z));

	// world space to parameter space
	Matrix44 Q = T*S;
	Matrix44 QInv = SInv*TInv;

	glUseProgram(g_solidShader);
	glPushMatrix();
	glTranslatef(-radius, 0.0f, 0.0f);	
	glScalef(1.0, 1.0f, 1.0);
	glutSolidSphere(radius, 20, 20);
	glPopMatrix();

	const float viewHeight = tanf(DegToRad(fov)/2.0f);

	glUseProgram(g_pointShader);
	glUniform1f( glGetUniformLocation(g_pointShader, "pointScale"), g_screenHeight/viewHeight);
	glUniform1f( glGetUniformLocation(g_pointShader, "pointRadius"), radius);
	glUniform3fv( glGetUniformLocation(g_pointShader, "invViewport"), 1, Vec3(1.0f/g_screenWidth, 1.0f/g_screenHeight, 1.0f));
	glUniform3fv( glGetUniformLocation(g_pointShader, "invProjection"), 1, Vec3(aspect*viewHeight, viewHeight, 1.0f));
	glUniformMatrix4fv( glGetUniformLocation(g_pointShader, "invQuadric"), 1, false, QInv);

	glDepthMask(GL_TRUE);
	glEnable(GL_DEPTH_TEST);		
	
/*	
	glEnable(GL_POINT_SPRITE);
	glTexEnvi(GL_POINT_SPRITE, GL_COORD_REPLACE, GL_TRUE);
	glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
	glBegin(GL_POINTS);
	glVertex3fv(quadricPos);
	glEnd();	
*/	

	
	// build billboard
	Matrix44 view, projection;
	glGetFloatv(GL_MODELVIEW_MATRIX, (float*)&view);
	glGetFloatv(GL_PROJECTION_MATRIX, (float*)&projection);

	// transform a normal to parameter space
	Matrix44 invClip = Transpose(projection*view*Q);

	// solve for the right hand bounds in homogenous clip space
	float a1 = DotInvW(invClip.columns[3], invClip.columns[3]);
	float b1 = -2.0f*DotInvW(invClip.columns[0], invClip.columns[3]);
	float c1 = DotInvW(invClip.columns[0], invClip.columns[0]); 

	float xmin, xmax;
 	SolveQuadratic(a1, b1, c1, xmin, xmax);	

	// solve for the right hand bounds in homogenous clip space
	float a2 = DotInvW(invClip.columns[3], invClip.columns[3]);
	float b2 = -2.0f*DotInvW(invClip.columns[1], invClip.columns[3]);
	float c2 = DotInvW(invClip.columns[1], invClip.columns[1]); 

	float ymin, ymax;
 	SolveQuadratic(a2, b2, c2, ymin, ymax);	

	Vec4 quadVertices[4] = 
	{
	   	Vec4(xmin, ymax, 0.0f, 1.0f),
	   	Vec4(xmin, ymin, 0.0f, 1.0f),
	   	Vec4(xmax, ymin, 0.0f, 1.0f),
	   	Vec4(xmax, ymax, 0.0f, 1.0f)
	};

	glBegin(GL_QUADS);
	glVertex3fv(quadVertices[0]);
	glVertex3fv(quadVertices[1]);
	glVertex3fv(quadVertices[2]);
	glVertex3fv(quadVertices[3]);
	glEnd();	

	glUseProgram(0);
	
	// flip
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
		case 'd':
		{
			break;
		}
		case ' ':
		{
			break;
		}
	}
}

static int lastx;
static int lasty;

void GLUTMouseFunc(int b, int state, int x, int y)
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
		}
	}
}

void GLUTMotionFunc(int x, int y)
{
	
    //int dx = x-lastx;
    //int dy = y-lasty;

	lastx = x;
	lasty = y;
	
}

void GLUTPassiveMotionFunc(int x, int y)
{
    //int dx = x-lastx;
    //int dy = y-lasty;
	
	lastx = x;
	lasty = y;

}


int main(int argc, char* argv[])
{	
    // init gl
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH);
	
    glutInitWindowSize(g_screenWidth, g_screenHeight);
    glutCreateWindow("Empty");
    glutPositionWindow(350, 100);
	
#if WIN32
	glewInit();
#endif

    Init();
	
    glutMouseFunc(GLUTMouseFunc);
    glutReshapeFunc(GLUTReshape);
    glutDisplayFunc(GLUTUpdate);
    glutKeyboardFunc(GLUTKeyboardDown);
    glutKeyboardUpFunc(GLUTKeyboardUp);
    glutIdleFunc(GLUTUpdate);	
    glutSpecialFunc(GLUTArrowKeys);
    glutSpecialUpFunc(GLUTArrowKeysUp);
    glutMotionFunc(GLUTMotionFunc);
	glutPassiveMotionFunc(GLUTPassiveMotionFunc);
	
    glutMainLoop();
}

