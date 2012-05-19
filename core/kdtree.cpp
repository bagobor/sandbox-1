#include "KdTree.h"

KdTree::KdTree(Point3* vertices, uint32 numVerts, uint32* indices, uint32 numFaces) 
    : m_vertices(vertices)
    , m_numVerts(numVerts)
    , m_indices(indices)
    , m_numFaces(numFaces)
{
    // build stats
    m_treeDepth = 0;
    m_innerNodes = 0;
    m_leafNodes = 0;

    Build();
}

void KdTree::CalculateFaceBounds(uint32* faces, uint32 numFaces, Vector3& outMinExtents, Vector3& outMaxExtents)
{
    Vector3 minExtents(FLT_MAX);
    Vector3 maxExtents(-FLT_MAX);

    // calculate face bounds
    for (uint32 i=0; i < numFaces; ++i)
    {
        Vector3 a = m_vertices[m_indices[faces[i]*3+0]];
        Vector3 b = m_vertices[m_indices[faces[i]*3+1]];
        Vector3 c = m_vertices[m_indices[faces[i]*3+2]];

        minExtents = Min(a, minExtents);
        maxExtents = Max(a, maxExtents);

        minExtents = Min(b, minExtents);
        maxExtents = Max(b, maxExtents);

        minExtents = Min(c, minExtents);
        maxExtents = Max(c, maxExtents);
    }

    outMinExtents = minExtents;
    outMaxExtents = maxExtents;
}

// helpers
namespace
{
	inline uint32 LongestAxis(const Vector3& v)
	{    
		if (v.x > v.y && v.x > v.z)
			return 0;
		else
			return (v.y > v.z) ? 1 : 2;
	}

} // namespace anonymous

void KdTree::Build()
{
   assert(m_numFaces*3);

    const double startTime = GetSeconds();

    const uint32 numFaces = m_numFaces;

    // build initial list of faces
    m_faces.resize(numFaces);
    for (uint32 i=0; i < numFaces; ++i)
    {
        m_faces[i] = i;
    }

    // calculate bounds of each face and store
    m_faceBounds.resize(numFaces);   
    for (uint32 i=0; i < m_faces.size(); ++i)
    {
        CalculateFaceBounds(&m_faces[i], 1, m_faceBounds[i].m_min, m_faceBounds[i].m_max);
    }

	m_nodes.reserve(numFaces*1.5f);

    // allocate space for all the nodes
	m_freeNode = 1;

    // start building
    BuildRecursive(0, &m_faces[0], numFaces);

    assert(s_depth == 0);

    const double buildTime = (GetSeconds()-startTime);

    cout << "KdTree Build Stats:" << endl;
    cout << "Node size: " << sizeof(Node) << endl;
    cout << "Build time: " << buildTime << "s" << endl;
    cout << "Inner nodes: " << m_innerNodes << endl;
    cout << "Leaf nodes: " << m_leafNodes << endl;
    cout << "Alloc nodes: " << m_nodes.size() << endl;
    cout << "Avg. tris/leaf: " << m_faces.size() / float(m_leafNodes) << endl;
    cout << "Max depth: " << m_treeDepth << endl;

    // free some memory
    FaceBoundsArray f;
    m_faceBounds.swap(f);
}

float KdTree::PartitionMedian(Node& n, uint32* faces, uint32 numFaces)
{


	uint32 mid = numFaces / 2;
	return m_faceBounds[faces[mid]].m_min.

}

float KdTree::PartitionSAH(Node& n, uint32* faces, uint32 numFaces)
{
	return 0;
}

void KdTree::BuildRecursive(uint32 nodeIndex, uint32* faces, uint32 numFaces)
{	    
    // if we've run out of nodes allocate some more
    if (nodeIndex >= m_nodes.size())
    {
		uint32 s = std::max(uint32(1.5f*m_nodes.size()), 512U);
        m_nodes.resize(s);
    }

	Node& n = m_nodes[nodeIndex];

	// pick axis and split point
	Vector3 nmin, nmax;
	CalculateFaceBounds(faces, numFaces, nmin, nmax);

	if (numFaces < 6)
	{
		// create leaf node

	}
	else
	{
		// create inner nodes and partition
		Vector3 edges = nmax-nmin;
		
		uint32 axis = LongestAxis(edges);
		float split = edges[axis]*0.5f;

		n.m_axis = axis;
		n.m_split = split;
		n.m_firstChild = nodeIndex+1;
		
		FaceArray leftFaces;
		FaceArray rightFaces;

		BuildRecursive(n.m_firstChild, 
	}
}


bool KdTree::TraceRay(const Point3& start, const Vector3& dir, float& outT, float& u, float& v, float& w, float& faceSign, uint32& faceIndex) const
{

}