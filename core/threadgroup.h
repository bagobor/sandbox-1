#pragma once

#include "types.h"
#include "threading.h"

#include <vector>

typedef uint32 (*ThreadFunc)(void*);

// very simple platform independent threading wrapper
class ThreadPool
{
public:

	ThreadPool();
    ~ThreadPool();

	void AddTask(ThreadFunc function, void* param);

	void Run(uint32 workerCount);
	void Wait();

	uint32 GetNumTasks() { return m_tasks.size(); }

private:

    static DWORD WINAPI WorkerMain(void* data);

    void AddWorker();

    struct Task
    {
        ThreadFunc m_func;
        void* m_param;
    };

    CriticalSection m_mutex;
    
    std::vector<Task> m_tasks;
	std::vector<HANDLE> m_threads;
};
