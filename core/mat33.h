#pragma once

#include "maths.h"

struct Matrix33
{
	Matrix33() {}
	Matrix33(const Vec3& c1, const Vec3& c2, const Vec3& c3)
	{
		cols[0] = c1;
		cols[1] = c2;
		cols[2] = c3;
	}

	float operator()(int i, int j) const { return static_cast<const float*>(cols[j])[i]; }
	float& operator()(int i, int j) { return static_cast<float*>(cols[j])[i]; }

	Vec3 cols[3];

	static inline Matrix33 Identity() { static const Matrix33 sIdentity(Vec3(1.0f, 0.0f, 0.0f), Vec3(0.0f, 1.0f, 0.0f), Vec3(0.0f, 0.0f, 1.0f)); return sIdentity; }
};

inline Matrix33 Multiply(float s, const Matrix33& m)
{
	Matrix33 r = m;
	r.cols[0] *= s;
	r.cols[1] *= s;	
	r.cols[2] *= s;
	return r;
}

inline Vec3 Multiply(const Matrix33& a, const Vec3& x)
{
	return a.cols[0]*x.x + a.cols[1]*x.y + a.cols[2]*x.z;
}
inline Vec3 operator*(const Matrix33& a, const Vec3& x) { return Multiply(a, x); }


inline Matrix33 Multiply(const Matrix33& a, const Matrix33& b)
{
	Matrix33 r;
	r.cols[0] = a*b.cols[0];
	r.cols[1] = a*b.cols[1];
	r.cols[2] = a*b.cols[2];
	return r;
}

inline Matrix33 Add(const Matrix33& a, const Matrix33& b)
{
	return Matrix33(a.cols[0]+b.cols[0], a.cols[1]+b.cols[1], a.cols[2]+b.cols[2]);
}

inline float Determinant(const Matrix33& m)
{
	return Dot(m.cols[0], Cross(m.cols[1], m.cols[2]));
}

inline Matrix33 Transpose(const Matrix33& a)
{
	Matrix33 r;
	for (uint32_t i=0; i < 3; ++i)
		for(uint32_t j=0; j < 3; ++j)
			r(i, j) = a(j, i);

	return r;
}

inline float Trace(const Matrix33& a)
{
	return a(0,0)+a(1,1)+a(2,2);
}

inline Matrix33 operator*(float s, const Matrix33& a) { return Multiply(s, a); }
inline Matrix33 operator*(const Matrix33& a, float s) { return Multiply(s, a); }
inline Matrix33 operator*(const Matrix33& a, const Matrix33& b) { return Multiply(a, b); }
inline Matrix33 operator+(const Matrix33& a, const Matrix33& b) { return Add(a, b); }
inline Matrix33 operator-(const Matrix33& a, const Matrix33& b) { return Add(a, -1.0f*b); }
inline Matrix33& operator+=(Matrix33& a, const Matrix33& b) { a = a+b; return a; }
inline Matrix33& operator-=(Matrix33& a, const Matrix33& b) { a = a-b; return a; }


