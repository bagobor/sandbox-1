#pragma once

#include "Core/Core.h"
#include "Core/Types.h"

#include <cstring>
#include <cassert>

#include <map>
#include <string>
#include <ostream>

uint32 memcrc32(const void* s, uint32 len);

// case insensitive string crc
uint32 stricrc32(const char* s);

// hash taken from the Linux kernel
uint32 strhash(const char* s);
uint32 strhash(const char* s, uint32 len);

// helper for use with unordered_* containers
template <typename T>
class MemoryHash
{
public:

	size_t operator()(const T& data) const
	{
		return memcrc32(&data, sizeof(T));
	}
};

// this can be replaced by a crc eventually
class NameHash
{
public:

	NameHash() : m_hash(0) {}
	NameHash(const char* name) 
	{
		assert(name);
		m_hash = strhash(name);

		AddName(m_hash, name);
	}

	uint32 GetHash() const { return m_hash; }
	const char* ToString() const {	return m_hash?FindName(m_hash):""; }

	bool operator==(const NameHash& rhs) const
	{		
		return m_hash == rhs.m_hash;
		///return strcmp(m_name, key.m_name) == 0;
	}

	bool operator<(const NameHash& rhs) const
	{
		return m_hash < rhs.m_hash;
	}

	static uint32 StaticHash(const char* s) { return strhash(s); }
	static uint32 StaticHash(const char* s, uint32 len) { return strhash(s, len); }

private:
	
	unsigned long m_hash;

	// all hashed strings
	typedef std::map<uint32, std::string> NameMap;

	static const char* FindName(uint32 hash)
	{
		NameMap::iterator iter = GetMap().find(hash);

		if (iter != GetMap().end())
		{
			return iter->second.c_str();
		}
		else
		{
			return NULL;
		}
	}

	// interns the string and checks for collisions, returns ptr to copy
	static void AddName(uint32 hash, const char* str)
	{		
		const char* s = FindName(hash);

		if (s)
		{
			// collision check
			assert(stricmp(str, s) == 0);
		}		
		else
		{
			GetMap().insert(make_pair(hash, std::string(str)));
		}
	}

	static NameMap& GetMap()
	{
		static NameMap s_names;
		return s_names;
	}
};

typedef NameHash Name;

inline std::ostream& operator << (std::ostream& s, const Name& n)
{
	return s << n.ToString();
}

inline std::istream& operator >> (std::istream& s, Name& n)
{
	std::string str;
	s >> str;
	n = Name(str.c_str());

	return s;
}
