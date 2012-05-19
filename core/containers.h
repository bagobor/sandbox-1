#pragma once

// an intrusive doubly linked list, T is required to have a m_next and m_prev ptrs
// warning: unfinished

template <typename T>
class IntrusiveDList
{
public:

	IntrusiveDList() : m_head(NULL)
	{

	}

	class iterator
	{
	public:

		iterator(T* e) : m_elem(e) {}

		T* operator *() const
		{
			return m_elem;
		}

		T& operator->() const
		{
			return *m_elem;	
		}

		void operator++() 
		{
			m_elem = m_elem->m_next;
		}

		void operator++(int)
		{
			++(*this);
		}

		void operator--()
		{
			m_elem = m_elem->m_prev;
		}

		void operator --(int)
		{
			--(*this);
		}

		T* m_elem;
	};

	iterator begin()
	{
		return iterator(m_head);
	}

	void push_back(T* elem)
	{
		elem->m_next = m_head;
		elem->m_prev = NULL;

		if (m_head)
			m_head->m_prev = elem;

		m_head = elem;
	}

	void erase(T* elem)
	{
		// update the previous element (or the head if there is none)
		if (elem->m_prev)
			elem->m_prev->m_next = elem->m_next;
		else
			m_head = elem->m_next;

		// update the next element's previous ptr
		if (elem->m_next)
			elem->m_next->m_prev = elem->m_prev;
	}

	virtual ~IntrusiveDList()
	{
	}

	T* m_head;
};