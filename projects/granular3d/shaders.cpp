#include "shaders.h"

#include <core/types.h>
#include <core/maths.h>
#include <core/shader.h>

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
