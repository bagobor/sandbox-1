#pragma once

#include "Core/Core.h"
#include "Core/Maths.h"

#include "Scene.h"

Colour PathTrace(const Scene& s, const Point3& rayOrigin, const Vector3& rayDir);
Colour ForwardTraceImportance(const Scene& scene, const Point3& startOrigin, const Vector3& startDir);
Colour ForwardTraceUniform(const Scene& scene, const Point3& startOrigin, const Vector3& startDir);
Colour Whitted(const Scene& s, const Point3& rayOrigin, const Vector3& rayDir);
Colour Debug(const Scene& s, const Point3& rayOrigin, const Vector3& rayDir);

