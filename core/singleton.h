#pragma once

#include <assert.h>

template <typename T>
class ISingleton
{
protected:

	//! prohibit copying, unfortunately I have had to implement these as I can't get SqPlus to bind them otherwise
	//! the best I can do currently is assert.
	ISingleton(const ISingleton&) { assert(0); }
	ISingleton& operator=(const ISingleton&) { assert(0); return *new ISingleton();}	

	ISingleton() {};
	virtual ~ISingleton() {};
	
	static T* mInstance;
	
public:
	
	static void Create()
	{
		//! don't try and call create twice
		assert(mInstance == 0);
		mInstance = new T();
	}
	
	static void Destroy()
	{
		assert(mInstance != 0);
		delete mInstance;
		mInstance = 0;
	}
	
	static inline T& Get()
	{
		//! must call create first
		assert(mInstance != 0);
		return *mInstance;
	}
};

//! helper macro for definining singleton linker symbol
#define IMPLEMENT_SINGLETON(c) template<> c* c::mInstance=0;