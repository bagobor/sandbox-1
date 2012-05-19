#pragma once

#include "Types.h"			

#include <cassert>

inline void* AlignedMalloc(size_t size,int byteAlign)
{
    void *mallocPtr = malloc(size + byteAlign + sizeof(void*));
    size_t ptrInt = (size_t)mallocPtr;

    ptrInt = (ptrInt + byteAlign + sizeof(void*)) / byteAlign * byteAlign;
    *(((void**)ptrInt) - 1) = mallocPtr;

    return (void*)ptrInt;
}

inline void AlignedFree(void *ptr)
{
    free(*(((void**)ptr) - 1));
}


// temporary allocation storage, used to reduce load on crt malloc
class MemoryArena
{
public:

	MemoryArena(const uint32 sizeInBytes)
	{
		m_mem = (byte*)AlignedMalloc(sizeInBytes, 16);
		m_size = sizeInBytes;
		m_head = m_mem;
	}

	~MemoryArena()
	{
		AlignedFree(m_mem);
	}

	byte* Allocate(uint32 size)
	{
		if ((m_head+size)-m_mem > ptrdiff(m_size))
		{
			assert(!"Arena ran out of memory");
			return NULL;
		}

		byte* p = m_head;
		m_head += size;

		return p;
	}
	
	void Reset()
	{
		m_head = m_mem;
	}

	byte* m_mem;
	byte* m_head;
	uint32 m_size;
};

// version of placement new to support usage as: new (MemArena) Type(); 
inline void* operator new (size_t s, MemoryArena& a)
{
	return a.Allocate(s);
}

// not used but compiler will complain without it
inline void operator delete(void*, MemoryArena&)
{
}
