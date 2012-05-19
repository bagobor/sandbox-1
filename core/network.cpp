#include "Network.h"
#include "Log.h"

using namespace std;

NetId GenerateNetId()
{
	// super shitty 'unique' identification
	return (rand() << 15) + rand();
}

IMPLEMENT_SINGLETON(Network);

Network::Network() : m_host(NULL)
{
	// initialize network handlers
	memset(m_msgHandlers, 0, sizeof(m_msgHandlers));

	if (enet_initialize () != 0)
	{
		Log::Info << "Could not initialize ENet" << std::endl;
	}

	atexit (enet_deinitialize);
}

Network::~Network()
{
	if (m_host)
		enet_host_destroy(m_host);
}

void Network::StartServer(uint16 port)
{
	// make sure we dont have a host already started
	assert(!m_host);

	// create host
	ENetAddress address;
	ENetHost * host;

	/* Bind the server to the default localhost.     */
	/* A specific host address can be specified by   */
	/* enet_address_set_host (& address, "x.x.x.x"); */

	address.host = ENET_HOST_ANY;
	/* Bind the server to port specified. */
	address.port = port;

	host = enet_host_create (& address /* the address to bind the server host to */, 
		32      /* allow up to 32 clients and/or outgoing connections */,
		0      /* assume any amount of incoming bandwidth */,
		0      /* assume any amount of outgoing bandwidth */);
	if (host == NULL)
	{
		Log::Info << "Network: An error occurred while trying to create an ENet server host." << std::endl;
		exit (EXIT_FAILURE);
	}
	else
		Log::Info << "Network: Host started on port " << port << std::endl;

	m_host = host;
}

void Network::Update(float dt)
{
	UNUSED(dt);

	ENetEvent event;

	/* Wait up to 1000 milliseconds for an event. */
	while (enet_host_service (m_host, & event, 0) > 0)
	{
		switch (event.type)
		{
		case ENET_EVENT_TYPE_CONNECT:
			char buf[255];
			enet_address_get_host_ip(&event.peer->address,buf, 255);

			printf ("A new client connected from %s:%u.\n", 
				buf,
				event.peer -> address.port);

			/* Store any relevant client information here. */
			event.peer -> data = "Client information";

			break;

		case ENET_EVENT_TYPE_RECEIVE:
			printf ("A packet of length %u containing %s was received from %s on channel %u.\n",
				event.packet -> dataLength,
				event.packet -> data,
				event.peer -> data,
				event.channelID);

			// unpack and process
			ProcessPacket((byte*)event.packet->data, event.packet->dataLength);

			/* Clean up the packet now that we're done using it. */
			enet_packet_destroy (event.packet);

			break;

		case ENET_EVENT_TYPE_DISCONNECT:
			printf ("%s disconected.\n", event.peer -> data);

			/* Reset the peer's client information. */

			event.peer -> data = NULL;
		}
	}
}

void Network::BroadcastData(void* data, size_t length)
{
	ENetPacket * packet = enet_packet_create (data, length, ENET_PACKET_FLAG_RELIABLE);

	/* Extend the packet so and append the string "foo", so it now */
	/* contains "packetfoo\0"                                      */
//	enet_packet_resize (packet, strlen ("packetfoo") + 1);
//	strcpy (& packet -> data [strlen ("packet")], "foo");

	/* Send the packet to the peer over channel id 0. */
	/* One could also broadcast the packet by         */
	/* enet_host_broadcast (host, 0, packet);         */
	//enet_peer_send (peer, 0, packet);

	enet_host_broadcast (m_host, 0, packet); 
		
	/* One could just use enet_host_service() instead. */
	//enet_host_flush (host);
}

void Network::BroadcastMsg(const NetMsg& msg)
{
	ENetPacket* packet = enet_packet_create(msg.stream.GetData(), msg.stream.GetSize(), ENET_PACKET_FLAG_RELIABLE);

	enet_host_broadcast (m_host, 0, packet); 
}

ENetPeer* Network::Connect(const char* a, uint16 port)
{
	ENetAddress address;
	ENetEvent event;
	ENetPeer *peer;

	/* Connect to some.server.net:1234. */
	enet_address_set_host (&address, a);
	address.port = port;

	/* Initiate the connection, allocating the two channels 0 and 1. */
	peer = enet_host_connect (m_host, & address, 2);    

	if (peer == NULL)
	{
		fprintf (stderr, "No available peers for initiating an ENet connection.\n");
		exit (EXIT_FAILURE);
	}

	/* Wait up to 5 seconds for the connection attempt to succeed. */
	if (enet_host_service (m_host, & event, 5000) > 0 &&
		event.type == ENET_EVENT_TYPE_CONNECT)
	{
		// add to connected peers list
		m_peers.push_back(peer);

		Log::Info << "Network: Successfully connected to " << a << ":" << port << endl;
	}
	else
	{
		/* Either the 5 seconds are up or a disconnect event was */
		/* received. Reset the peer in the event the 5 seconds   */
		/* had run out without any significant event.            */
		enet_peer_reset (peer);

		Log::Info << "Network: Connection to " << a << ":" << port << " failed" << endl;
	}

	return peer;
}

// could seperate this out better, the only place it's used is in the message dispatch stuff
//#include "Game.h"

//extern Game* gGame;

void Network::ProcessPacket(byte* data, size_t len)
{
	UNUSED(data);
	UNUSED(len);
	/*
	// read msg id and handle it
	if (len < sizeof(MsgTypeId))
	{
		Log::Info << "Network: Malformed network message." << endl;
	}
	else
	{
		NetStream stream(data, len);

		while (stream.GetDataRemaining())
		{
			// read out the message
			MsgTypeId id;
			stream >> id;

			// invalid msg id
			assert(id >= 0 && id < kMsgMax);

			Log::Info << "Network: Processing message of type: " << id << endl;

			// big 'ol switch statement
			switch (id)
			{
			case kMsgPlayerJoinRequest:
				gGame->OnReceivePlayerJoinRequest(msg); break;
			case kMsgPlayerJoinResult:
				gGame->OnReceivePlayerJoinResult(msg); break;
			};
		}
	}
	*/
}

