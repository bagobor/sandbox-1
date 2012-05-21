#pragma once

#include "singleton.h"
#include "core.h"
#include "external/enet-1.2/include/enet/enet.h"

#include <vector>

// handles bit packing 
class NetStream
{
public:

	NetStream() : m_buffer(new byte[1024]), m_length(1024), m_pos(0), m_freeMemory(true)
	{
		
	}

	~NetStream()
	{
		if (m_freeMemory)
			delete[] m_buffer;
	}

	NetStream(void* memory, size_t len) : m_buffer((byte*)memory),
										  m_length(len),
										  m_pos(0),
										  m_freeMemory(false) 
	{}

	template <typename T>
	NetStream& operator << (const T& p)
	{
		assert(m_pos + sizeof(T) < m_length);

		memcpy(&m_buffer[m_pos], &p, sizeof(T));
		m_pos += sizeof(T);

		return *this;
	}

	template <typename T>
	NetStream& operator >> (T& p)
	{
		assert(m_pos + sizeof(T) < m_length);

		memcpy(&p, &m_buffer[m_pos], sizeof(T));
		m_pos += sizeof(T);

		return *this;
	}

	void WriteString(const char* s)
	{
		assert(s);

		do
		{
			m_buffer[m_pos] = *s;
			++m_pos;
		} 
		while (*s);
	}

	const char* ReadString()
	{
		const char* s = reinterpret_cast<const char*>(m_buffer);

		while (m_buffer[m_pos] && GetDataRemaining())
			++m_pos;

		// if no null terminator there was an error
		assert(m_buffer == 0);
		++m_pos;

		return s;
	}

	void* GetData() const { return m_buffer; }
	size_t GetSize() const { return m_pos; }

	size_t GetDataRemaining() const { return m_length - m_pos - 1; }

private:
	
	byte* m_buffer;
	size_t m_length;
	size_t m_pos;

	bool m_freeMemory;	
};

// kind of unique id
typedef size_t NetId;
NetId GenerateNetId();

// message types
enum MessageType
{
	kMsgPlayerJoinRequest,
	kMsgPlayerJoinResult,
	kMsgPlayerLeave,
	kMsgCreateActor,
	kMsgDestroyActor,
	kMsgUpdateActor,
	kMsgMax
};

class NetMsg;

// msg id
typedef int MsgTypeId;

// msg handler
typedef void (*MessageHandlerFunc)(NetStream& stream);

// very thin wrapper on Enet
class Network : public ISingleton<Network>
{
public:

	Network();
	~Network();

	void StartServer(uint16 listenport);
	void StopServer();
	
	// connect to a remote peer
	ENetPeer* Connect(const char* address, uint16 port);

	// tick networking
	void Update(float dt);

	// send to one peer
	void SendMsg(ENetPeer* p, const NetMsg& m);

	// sends a msg to all peers
	void BroadcastMsg(const NetMsg& m);
	void BroadcastData(void* data, size_t len);

	// set handler for a message type
	void SetMsgHandler(MessageType m, MessageHandlerFunc handler) { m_msgHandlers[m] = handler; }

private:

	void ProcessPacket(byte* data, size_t len);
	
	// our local host
	ENetHost* m_host;

	// connected peers
	typedef std::vector<ENetPeer*> PeerArray;
	PeerArray m_peers;

	// array of message handlers
	MessageHandlerFunc m_msgHandlers[kMsgMax];
};


class NetMsg
{
public:

	NetMsg(MessageType m)
	{
		stream << m;
	}

	// helper function
	void Send(ENetPeer* peer)
	{
		Network::Get().SendMsg(peer, *this);
	}

	// who sent this message
	ENetPeer* sender;

	// chunk of data
	NetStream stream;
};

