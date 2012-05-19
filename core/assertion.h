#pragma once

#ifndef NDEBUG

// undefine existing assert
#ifdef ASSERT
#undef ASSERT
#endif

#include "Types.h"

//! custom assert function
extern bool CustomAssert(const tchar *exp, const tchar *msg, int line, const char *file, bool& ignoreAlways);

//! our custom assert macro
//! has nifty ignore always functionality, a better message box and actually breaks into the code properly!
#define ASSERT(x) \
	do	\
	{	\
		static bool ignoreAlways = false;	\
		if (!ignoreAlways)	\
			if (!(x) && CustomAssert(_TS(#x), NULL, __LINE__, __FILE__, ignoreAlways))	\
				_asm { int 3 };		\
	}	\
	while (0)

//! as above but allows the specification of a message to put on the assertion box
#define ASSERTM(x, msg)	\
	do	\
	{	\
		static bool ignoreAlways = false;	\
		if (!ignoreAlways)	\
			if (!(x) && CustomAssert(_TS(#x), msg, __LINE__, __FILE__, ignoreAlways))		\
				_asm { int 3 };		\
	}	\
	while (0)

#else

//! disable assert macro
#define ASSERT(x) ((void)0)
#define ASSERTM(x, desc) ((void)0)

#endif	// NDEBUG

//! static assertion
//! uses same idea as Boost's one: taking sizeof an incomplete type will generate a compiler error
template <bool>
struct STATIC_ASSERTION_FAILED;

template <>
struct STATIC_ASSERTION_FAILED<true> { };

template <int>
struct static_assertion_test { };

//! allows us to use ## when one of the arguments is a macro
#define JOIN(X, Y) DO_JOIN(X, Y)
#define DO_JOIN(X, Y) DO_JOIN2(X, Y)
#define DO_JOIN2(X, Y) X##Y

//! only use once per line
//! have to use __COUNTER__ because __LINE__ is broken if Edit and Continue mode is enabled (go figure ...) see Q199057
#define STATIC_ASSERT(exp) \
	typedef jl::static_assertion_test< sizeof(jl::STATIC_ASSERTION_FAILED<(bool)(exp)>) > JOIN(static_assert_, __COUNTER__)

