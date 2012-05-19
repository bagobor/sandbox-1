#pragma once

#include <Core/Maths.h>

struct Matrix22
{
	Matrix22() {}
	Matrix22(float a, float b, float c, float d)
	{
		cols[0] = Vec2(a, c);
		cols[1] = Vec2(b, d);
	}

	Matrix22(const Vec2& c1, const Vec2& c2)
	{
		cols[0] = c1;
		cols[1] = c2;
	}

	float operator()(int i, int j) const { return static_cast<const float*>(cols[j])[i]; }
	float& operator()(int i, int j) { return static_cast<float*>(cols[j])[i]; }

	Vec2 cols[2];

	static inline Matrix22 Identity() { static const Matrix22 sIdentity(Vec2(1.0f, 0.0f), Vec2(0.0f, 1.0f)); return sIdentity; }
};

inline Matrix22 Multiply(float s, const Matrix22& m)
{
	Matrix22 r = m;
	r.cols[0] *= s;
	r.cols[1] *= s;	
	return r;
}

inline Matrix22 Multiply(const Matrix22& a, const Matrix22& b)
{
	Matrix22 r;
	r.cols[0] = a.cols[0]*b.cols[0].x + a.cols[1]*b.cols[0].y;
	r.cols[1] = a.cols[0]*b.cols[1].x + a.cols[1]*b.cols[1].y;
	return r;
}

inline Vec2 Multiply(const Matrix22& a, const Vec2& x)
{
	return a.cols[0]*x.x + a.cols[1]*x.y;
}

inline Matrix22 Add(const Matrix22& a, const Matrix22& b)
{
	return Matrix22(a.cols[0]+b.cols[0], a.cols[1]+b.cols[1]);
}

inline float Determinant(const Matrix22& m)
{
	return m(0,0)*m(1,1)-m(1,0)*m(0,1);
}

inline Matrix22 Inverse(const Matrix22& m, float& det)
{
	det = Determinant(m); 

	if (fabsf(det) > FLT_EPSILON)
	{
		Matrix22 inv;
		inv(0,0) =  m(1,1);
		inv(1,1) =  m(0,0);
		inv(0,1) = -m(0,1);
		inv(1,0) = -m(1,0);

		return Multiply(1.0f/det, inv);	
	}
	else
	{
		det = 0.0f;
		return m;
	}
}

inline Matrix22 Transpose(const Matrix22& a)
{
	Matrix22 r;
	r(0,0) = a(0,0);
	r(0,1) = a(1,0);
	r(1,0) = a(0,1);
	r(1,1) = a(1,1);
	return r;
}

inline float Trace(const Matrix22& a)
{
	return a(0,0)+a(1,1);
}

inline Matrix22 RotationMatrix(float theta)
{
	return Matrix22(Vec2(cosf(theta), sinf(theta)), Vec2(-sinf(theta), cosf(theta)));
}

// outer product of a and b, b is considered a row vector
inline Matrix22 Outer(const Vec2& a, const Vec2& b)
{
	return Matrix22(a*b.x, a*b.y);
}

inline Matrix22 operator*(float s, const Matrix22& a) { return Multiply(s, a); }
inline Matrix22 operator*(const Matrix22& a, float s) { return Multiply(s, a); }
inline Matrix22 operator*(const Matrix22& a, const Matrix22& b) { return Multiply(a, b); }
inline Matrix22 operator+(const Matrix22& a, const Matrix22& b) { return Add(a, b); }
inline Matrix22 operator-(const Matrix22& a, const Matrix22& b) { return Add(a, -1.0f*b); }
inline Matrix22& operator+=(Matrix22& a, const Matrix22& b) { a = a+b; return a; }
inline Matrix22& operator-=(Matrix22& a, const Matrix22& b) { a = a-b; return a; }

inline Vec2 operator*(const Matrix22& a, const Vec2& x) { return Multiply(a, x); }


