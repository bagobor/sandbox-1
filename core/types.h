#pragma once

#include <cstddef>

#include <stdint.h>

#ifndef STDINT
#define STDINT

#ifdef WIN32

//! generic byte data
typedef unsigned char	byte;

//! unsigned types
typedef unsigned char	uchar;
typedef unsigned short	ushort;
typedef unsigned int	uint;
typedef unsigned long	ulong;

typedef signed char int8;
typedef short int16;
typedef int int32;
typedef __int64 int64;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned __int64 uint64;

//! an unsigned integer type large enough to hold a pointer
typedef intptr_t		intptr;
typedef uintptr_t		uintptr;
typedef ptrdiff_t		ptrdiff;

#else // For LINUX and MAC_TIGER

//! generic byte data
typedef unsigned char	byte;

//! unsigned types
typedef unsigned char	uchar;
typedef unsigned short	ushort;
typedef unsigned int	uint;
typedef unsigned long	ulong;

typedef int8_t int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;
typedef uint8_t uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

//! an unsigned integer type large enough to hold a pointer
typedef intptr_t		intptr;
typedef uintptr_t		uintptr;
typedef ptrdiff_t		ptrdiff;


#endif  // WIN32

// aliases
typedef uint8 u8;
typedef uint16 u16;
typedef uint32 u32;
typedef uint64 u64;


#endif // STDINT


#include <string>
#include <sstream>

#ifdef UNICODE

typedef wchar_t			tchar;
typedef std::wstring	tstring;
typedef std::wstringstream tstringstream;

#define _TS(x) L ## x

#else

typedef char			tchar;
typedef std::string		tstring;
typedef std::stringstream tstringstream;

#define _TS(x) x

#endif


// temp adding these for fmod compatibility which 
// seems to use them but not define them
#ifndef FALSE
#define FALSE               0
#endif

#ifndef TRUE
#define TRUE                1
#endif
