#pragma once

#include <pthread.h>

class Thread
{
public:

	pthread_t m_handle;

	Thread(ThreadFunc func, void* param)
	{
		int retval = pthread_create(&m_handle, NULL, func, param);
		assert(retval);		
	}
};

void WaitForThreads(const Thread* threads, uint32 n)
{
	// just join all the threads
	for (uint32 i=0; i < n; ++i)
	{
		pthread_join(threads[i], NULL);
	}
}
