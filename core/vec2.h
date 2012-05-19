#pragma once

#if _WIN32
#ifdef _DEBUG

#define VEC2_VALIDATE() {	assert(_finite(x));\
	assert(!_isnan(x));\
	\
	assert(_finite(y));\
	assert(!_isnan(y));\
						 }
#else

#define VEC2_VALIDATE() {\
	assert(isfinite(x));\
	assert(isfinite(y)); }\

#endif // _WIN32

#else
#define VEC2_VALIDATE()
#endif

#ifdef _DEBUG
#define FLOAT_VALIDATE(f) { assert(_finite(f)); assert(!_isnan(f)); }
#else
#define FLOAT_VALIDATE(f)
#endif


// vec2
template <typename T>
class XVector2
{
public:

	typedef T value_type;

	XVector2(float _x=0.0f, float _y=0.0f) : x(_x), y(_y) { VEC2_VALIDATE(); }
	XVector2(const float* p) : x(p[0]), y(p[1]) {}

	operator T* () { return &x; }
	operator const T* () const { return &x; };

	void Set(T x_, T y_) { VEC2_VALIDATE(); x = x_; y = y_; }

	XVector2<T> operator * (T scale) const { XVector2<T> r(*this); r *= scale; return r; VEC2_VALIDATE();}
	XVector2<T> operator / (T scale) const { XVector2<T> r(*this); r /= scale; return r; VEC2_VALIDATE();}
	XVector2<T> operator + (const XVector2<T>& v) const { XVector2<T> r(*this); r += v; return r; VEC2_VALIDATE();}
	XVector2<T> operator - (const XVector2<T>& v) const { XVector2<T> r(*this); r -= v; return r; VEC2_VALIDATE();}

	XVector2<T>& operator *=(T scale) {x *= scale; y *= scale; VEC2_VALIDATE(); return *this;}
	XVector2<T>& operator /=(T scale) {T s(1.0f/scale); x *= s; y *= s; VEC2_VALIDATE(); return *this;}
	XVector2<T>& operator +=(const XVector2<T>& v) {x += v.x; y += v.y; VEC2_VALIDATE(); return *this;}
	XVector2<T>& operator -=(const XVector2<T>& v) {x -= v.x; y -= v.y; VEC2_VALIDATE(); return *this;}

	XVector2<T>& operator *=(const XVector2<T>& scale) {x *= scale.x; y *= scale.y; VEC2_VALIDATE(); return *this;}

	// negate
	XVector2<T> operator -() const { VEC2_VALIDATE(); return XVector2<T>(-x, -y); }

	// returns this vector
	void Normalize() { *this /= Length(*this); }
	void SafeNormalize(const XVector2<T>& v=XVector2<T>(0.0f,0.0f))
	{
		T length = Length(*this);
		*this = (length==0.00001f)?v:(*this /= length);
	}

	T x;
	T y;
};

typedef XVector2<float> Vec2;
typedef XVector2<float> Vector2;

// lhs scalar scale
template <typename T>
XVector2<T> operator *(T lhs, const XVector2<T>& rhs)
{
	XVector2<T> r(rhs);
	r *= lhs;
	return r;
}

template <typename T>
XVector2<T> operator*(const XVector2<T>& lhs, const XVector2<T>& rhs)
{
	XVector2<T> r(lhs);
	r *= rhs;
	return r;
}

template <typename T>
bool operator==(const XVector2<T>& lhs, const XVector2<T>& rhs)
{
	return (lhs.x == rhs.x && lhs.y == rhs.y);
}


template <typename T>
T Dot(const XVector2<T>& v1, const XVector2<T>& v2)
{
	return v1.x * v2.x + v1.y * v2.y; 
}

// returns the ccw perpendicular vector 
template <typename T>
XVector2<T> PerpCCW(const XVector2<T>& v)
{
	return XVector2<T>(-v.y, v.x);
}

template <typename T>
XVector2<T> PerpCW(const XVector2<T>& v)
{
	return XVector2<T>(v.y, -v.x);
}

// component wise min max functions
template <typename T>
XVector2<T> Max(const XVector2<T>& a, const XVector2<T>& b)
{
	return XVector2<T>(Max(a.x, b.x), Max(a.y, b.y));
}

template <typename T>
XVector2<T> Min(const XVector2<T>& a, const XVector2<T>& b)
{
	return XVector2<T>(Min(a.x, b.x), Min(a.y, b.y));
}

// 2d cross product, treat as if a and b are in the xy plane and return magnitude of z
template <typename T>
T Cross(const XVector2<T>& a, const XVector2<T>& b)
{
	return (a.x*b.y - a.y*b.x);
}




