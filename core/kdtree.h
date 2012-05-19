#pragma once

class KdTree
{
public:

	KdTree(Point3* vertices, uint32 numVerts, uint32* indices, uint32 numFaces);

    bool TraceRaySlow(const Point3& start, const Vector3& dir, float& outT, Vector3* outNormal) const;
    bool TraceRay(const Point3& start, const Vector3& dir, float& outT, float& u, float& v, float& w, float& faceSign, uint32& faceIndex) const;

    void DebugDraw();
    
    Vector3 GetCenter() const { return (m_nodes[0].m_minExtents+m_nodes[0].m_maxExtents)*0.5f; }
    Vector3 GetMinExtents() const { return m_nodes[0].m_minExtents; }
    Vector3 GetMaxExtents() const { return m_nodes[0].m_maxExtents; }

private:

	// partition the objects and return the number of objects in the lower partition
	float PartitionMedian(Node& n, uint32* faces, uint32 numFaces);
	float PartitionSAH(Node& n, uint32* faces, uint32 numFaces);

    void Build();
    void BuildRecursive(uint32 nodeIndex, uint32* faces, uint32 numFaces);
    void TraceRecursive(uint32 nodeIndex, const Point3& start, const Vector3& dir, const Vector3& rcp_dir, float& outT, Vector3* outNormal) const;
 
    void CalculateFaceBounds(uint32* faces, uint32 numFaces, Vector3& outMinExtents, Vector3& outMaxExtents);
    uint32 GetNumFaces() const { return m_numFaces; }
	uint32 GetNumNodes() const { return m_nodes.size(); }

    // stats (reset each trace)
    static uint32 GetTraceDepth() { return s_traceDepth; }

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

	struct Node
	{
		Node()
			: m_axis(0)
			, m_position(0.0f)
			, m_faces(NULL)
			, m_numFaces(0)
		{
		}

		uint32 m_axis;
		float m_position;

		uint32* m_faces;

		union
		{
			uint32 m_numFaces;
			uint32 m_firstChild;	// node index of first child
		}
	};

	Vector m_minExtents;
	Vector3 m_maxExtents;

    const Point3* m_vertices;
    const uint32 m_numVerts;

    const uint32* m_indices;
    const uint32 m_numFaces;

	typedef std::vector<uint32> IndexArray;
    typedef std::vector<Point3> PositionArray;
    typedef std::vector<Node> NodeArray;
    typedef std::vector<uint32> FaceArray;
    typedef std::vector<Bounds> FaceBoundsArray;

	IndexArray m_indices;
	PositionArray m_positions;

	NodeArray m_nodes;

	FaceArray m_faces;
	FaceBoundsArray m_faceBounds;

    // stats
    uint32 m_treeDepth;
    uint32 m_innerNodes;
    uint32 m_leafNodes;   
};
