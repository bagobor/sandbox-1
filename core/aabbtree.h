#pragma once

#include "Core/Core.h"
#include "Core/Maths.h"

#include <vector>

class AABBTree
{
	AABBTree(const AABBTree&);
	AABBTree& operator=(const AABBTree&);

public:

    AABBTree(Point3* vertices, uint32 numVerts, uint32* indices, uint32 numFaces);

    bool TraceRaySlow(const Point3& start, const Vector3& dir, float& outT, Vector3* outNormal) const;
    bool TraceRay(const Point3& start, const Vector3& dir, float& outT, float& u, float& v, float& w, float& faceSign, uint32& faceIndex) const;

    void DebugDraw();
    
    Vector3 GetCenter() const { return (m_nodes[0].m_minExtents+m_nodes[0].m_maxExtents)*0.5f; }
    Vector3 GetMinExtents() const { return m_nodes[0].m_minExtents; }
    Vector3 GetMaxExtents() const { return m_nodes[0].m_maxExtents; }
	
#if _WIN32
    // stats (reset each trace)
    static uint32 GetTraceDepth() { return s_traceDepth; }
#endif
	
private:

    void DebugDrawRecursive(uint32 nodeIndex, uint32 depth);

    struct Node
    {
        Node() 	
            : m_faces(NULL)
            , m_numFaces(0)
            , m_minExtents(0.0f)
            , m_maxExtents(0.0f)
        {
        }

		union
		{
			uint32 m_children;
			uint32 m_numFaces;			
		};

		uint32* m_faces;        
        Vector3 m_minExtents;
        Vector3 m_maxExtents;
    };


    struct Bounds
    {
        Bounds() : m_min(0.0f), m_max(0.0f)
        {
        }

        Bounds(const Vector3& min, const Vector3& max) : m_min(min), m_max(max)
        {
        }

        inline float GetVolume() const
        {
            Vector3 e = m_max-m_min;
            return (e.x*e.y*e.z);
        }

        inline float GetSurfaceArea() const
        {
            Vector3 e = m_max-m_min;
            return 2.0f*(e.x*e.y + e.x*e.z + e.y*e.z);
        }

        inline void Union(const Bounds& b)
        {
            m_min = Min(m_min, b.m_min);
            m_max = Max(m_max, b.m_max);
        }

        Vector3 m_min;
        Vector3 m_max;
    };

    typedef std::vector<uint32> IndexArray;
    typedef std::vector<Point3> PositionArray;
    typedef std::vector<Node> NodeArray;
    typedef std::vector<uint32> FaceArray;
    typedef std::vector<Bounds> FaceBoundsArray;

	// partition the objects and return the number of objects in the lower partition
	uint32 PartitionMedian(Node& n, uint32* faces, uint32 numFaces);
	uint32 PartitionSAH(Node& n, uint32* faces, uint32 numFaces);

    void Build();
    void BuildRecursive(uint32 nodeIndex, uint32* faces, uint32 numFaces);
    void TraceRecursive(uint32 nodeIndex, const Point3& start, const Vector3& dir, float& outT, float& u, float& v, float& w, float& faceSign, uint32& faceIndex) const;
 
    void CalculateFaceBounds(uint32* faces, uint32 numFaces, Vector3& outMinExtents, Vector3& outMaxExtents);
    uint32 GetNumFaces() const { return m_numFaces; }
	uint32 GetNumNodes() const { return m_nodes.size(); }

	// track the next free node
	uint32 m_freeNode;

    const Point3* m_vertices;
    const uint32 m_numVerts;

    const uint32* m_indices;
    const uint32 m_numFaces;

    FaceArray m_faces;
    NodeArray m_nodes;
    FaceBoundsArray m_faceBounds;    

    // stats
    uint32 m_treeDepth;
    uint32 m_innerNodes;
    uint32 m_leafNodes; 
	
#if _WIN32
   _declspec (thread) static uint32 s_traceDepth;
#endif
};