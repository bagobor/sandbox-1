#pragma once

#include "core/maths.h"

#include <vector>

void Triangulate(const Vec2* points, uint32_t n, std::vector<Vec2>& outPoints, std::vector<uint32_t>& outTris);

void CreateTorus(std::vector<Vec2>& points, std::vector<uint32_t>& indices, float inner, float outer, uint32_t segments);


