#pragma once

#include "Maths.h"

class Bounds
{
public:
	
	Bounds() : m_sphereRadius(0.0f) {}
	Bounds(float radius) : m_sphereRadius(radius) {}

	inline void SetRadius(float r) {m_sphereRadius = r; }
	inline Vec3 GetCenter() const { return (m_minAABB + m_maxAABB) * 0.5f; }

	// min and max extents of the enclosing aabb
	Vec3 m_minAABB;
	Vec3 m_maxAABB;

	// the sphere radius
	float m_sphereRadius;
};

