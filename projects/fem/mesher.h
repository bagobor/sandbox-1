#pragma once

#include "core/maths.h"

#include <vector>

void TriangulateDelaunay(const Vec2* points, uint32_t numPoints, uint32_t maxPoints, float maxArea, float minAngle, std::vector<uint32_t>& outTris, std::vector<Vec2>& outPoints);
void CreateTorus(std::vector<Vec2>& points, std::vector<uint32_t>& indices, float inner, float outer, uint32_t segments);



