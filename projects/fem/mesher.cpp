#include "Mesher.h"

#include "Core/Maths.h"

#include <vector>

void Triangulate(const Vec2* points, uint32_t numPoints, const  std::vector<Vec2>& outPoints, std::vector<uint32_t>& outTris)
{
		

}

void CreateTorus(std::vector<Vec2>& points, std::vector<uint32_t>& indices, float inner, float outer, uint32_t segments)
{
	assert(inner < outer);

	for (uint32_t i=0; i < segments; ++i)
	{
		float theta = float(i)/segments*kPi*2.0f;
		
		float x = sinf(theta);
		float y = cosf(theta);
		
		points.push_back(Vec2(x, y)*outer);
		points.push_back(Vec2(x, y)*inner);

		if (i > 0)
		{
			uint32_t base = (i-1)*2;

			indices.push_back(base+0);
			indices.push_back(base+1);
			indices.push_back(base+2);

			indices.push_back(base+2);
			indices.push_back(base+1);
			indices.push_back(base+3);
		}
	}

	uint32_t base = points.size()-2;

	indices.push_back(base+0);
	indices.push_back(base+1);
	indices.push_back(0);

	indices.push_back(0);
	indices.push_back(base+1);
	indices.push_back(1);
}
