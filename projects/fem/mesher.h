#pragma once

#include "core/maths.h"
#include "core/tga.h"

#include <vector>

void TriangulateImage(const TgaImage& image, std::vector<Vec2>& outPoints, std::vector<uint32_t>& outTris);

void TriangulateDelaunay(const Vec2* points, uint32_t numPoints, std::vector<Vec2>& outPoints, std::vector<uint32_t>& outTris);
void TriangulateVariational(const Vec2* points, uint32_t numPoints, const Vec2* bpoints, uint32_t numBPoints, std::vector<Vec2>& outPoints, std::vector<uint32_t>& outTris);

void RefineDelaunay(const Vec2* points, uint32_t numPoints, const uint32_t* triangles, uint32_t numTris, uint32_t maxPoints, float minAngle, float maxArea, std::vector<Vec2>& outPoints, std::vector<uint32_t>& outTris);

void CreateTorus(std::vector<Vec2>& points, std::vector<uint32_t>& indices, float inner, float outer, uint32_t segments);




