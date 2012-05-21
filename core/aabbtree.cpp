#include "AABBTree.h"

#include "Maths.h"
#include "Platform.h"

#include <algorithm>
#include <iostream>

using namespace std;

#if _WIN32
_declspec (thread) uint32 AABBTree::s_traceDepth;
#endif

AABBTree::AABBTree(Point3* vertices, uint32 numVerts, uint32* indices, uint32 numFaces) 
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

namespace
{

	struct FaceSorter
	{
		FaceSorter(const Point3* positions, const uint32* indices, uint32 n, uint32 axis) 
			: m_vertices(positions)
			, m_indices(indices)
			, m_numIndices(n)
			, m_axis(axis)
		{        
		}

		inline bool operator()(uint32 lhs, uint32 rhs) const
		{
			float a = GetCentroid(lhs);
			float b = GetCentroid(rhs);

			if (a == b)
				return lhs < rhs;
			else
				return a < b;
		}

		inline float GetCentroid(uint32 face) const
		{
			const Point3& a = m_vertices[m_indices[face*3+0]];
			const Point3& b = m_vertices[m_indices[face*3+1]];
			const Point3& c = m_vertices[m_indices[face*3+2]];

			return (a[m_axis] + b[m_axis] + c[m_axis])/3.0f;
		}

		const Point3* m_vertices;
		const uint32* m_indices;
		uint32 m_numIndices;
		uint32 m_axis;
	};
	
	inline uint32 LongestAxis(const Vector3& v)
	{    
		if (v.x > v.y && v.x > v.z)
			return 0;
		else
			return (v.y > v.z) ? 1 : 2;
	}

} // anonymous namespace

void AABBTree::CalculateFaceBounds(uint32* faces, uint32 numFaces, Vector3& outMinExtents, Vector3& outMaxExtents)
{
    Vector3 minExtents(FLT_MAX);
    Vector3 maxExtents(-FLT_MAX);

    // calculate face bounds
    for (uint32 i=0; i < numFaces; ++i)
    {
        Vector3 a = Vector3(m_vertices[m_indices[faces[i]*3+0]]);
        Vector3 b = Vector3(m_vertices[m_indices[faces[i]*3+1]]);
        Vector3 c = Vector3(m_vertices[m_indices[faces[i]*3+2]]);

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

// track current tree depth
static uint32 s_depth = 0;

void AABBTree::Build()
{
    assert(m_numFaces*3);

    const double startTime = GetSeconds();

    const uint32 numFaces = m_numFaces;

    // build initial list of faces
    m_faces.reserve(numFaces);
	/*	
    for (uint32 i=0; i < numFaces; ++i)
    {
        m_faces[i] = i;
    }
	*/

    // calculate bounds of each face and store
    m_faceBounds.reserve(numFaces);   
    
	std::vector<Bounds> stack;
	for (uint32 i=0; i < numFaces; ++i)
    {
		Bounds top;
        CalculateFaceBounds(&i, 1, top.m_min, top.m_max);
		
		stack.push_back(top);

		while (!stack.empty())
		{
			Bounds b = stack.back();
			stack.pop_back();

			const float kAreaThreshold = 200.0f;

			if (b.GetSurfaceArea() < kAreaThreshold)
			{
				// node is good, append to our face list
				m_faces.push_back(i);
				m_faceBounds.push_back(b);
			}
			else
			{
				// split along longest axis
				uint32 a = LongestAxis(b.m_max-b.m_min);

				float splitPos = (b.m_min[a] + b.m_max[a])*0.5f;
				Bounds left(b);
				left.m_max[a] = splitPos;
				
				assert(left.GetSurfaceArea() < b.GetSurfaceArea());


				Bounds right(b);
				right.m_min[a] = splitPos;

				assert(right.GetSurfaceArea() < b.GetSurfaceArea());

				stack.push_back(left);				
				stack.push_back(right);
			}
		}
    }

	m_nodes.reserve(unsigned int(numFaces*1.5f));

    // allocate space for all the nodes
	m_freeNode = 1;

    // start building
    BuildRecursive(0, &m_faces[0], numFaces);

    assert(s_depth == 0);

    const double buildTime = (GetSeconds()-startTime);

    cout << "AAABTree Build Stats:" << endl;
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

// partion faces around the median face
uint32 AABBTree::PartitionMedian(Node& n, uint32* faces, uint32 numFaces)
{
	FaceSorter predicate(&m_vertices[0], &m_indices[0], m_numFaces*3, LongestAxis(n.m_maxExtents-n.m_minExtents));
    std::nth_element(faces, faces+numFaces/2, faces+numFaces, predicate);

	return numFaces/2;
}

// partion faces based on the surface area heuristic
uint32 AABBTree::PartitionSAH(Node& n, uint32* faces, uint32 numFaces)
{
	
	/*
    Vector3 mean(0.0f);
    Vector3 variance(0.0f);

    // calculate best axis based on variance
    for (uint32 i=0; i < numFaces; ++i)
    {
        mean += 0.5f*(m_faceBounds[faces[i]].m_min + m_faceBounds[faces[i]].m_max);
    }
    
    mean /= float(numFaces);

    for (uint32 i=0; i < numFaces; ++i)
    {
        Vector3 v = 0.5f*(m_faceBounds[faces[i]].m_min + m_faceBounds[faces[i]].m_max) - mean;
        v *= v;
        variance += v;
    }

    uint32 bestAxis = LongestAxis(variance);
	*/

	uint32 bestAxis = 0;
	uint32 bestIndex = 0;
	float bestCost = FLT_MAX;

	for (uint32 a=0; a < 3; ++a)	
	//uint32 a = bestAxis;
	{
		// sort faces by centroids
		FaceSorter predicate(&m_vertices[0], &m_indices[0], m_numFaces*3, a);
		std::sort(faces, faces+numFaces, predicate);

		// two passes over data to calculate upper and lower bounds
		vector<float> cumulativeLower(numFaces);
		vector<float> cumulativeUpper(numFaces);

		Bounds lower;
		Bounds upper;

		for (uint32 i=0; i < numFaces; ++i)
		{
			lower.Union(m_faceBounds[faces[i]]);
			upper.Union(m_faceBounds[faces[numFaces-i-1]]);

			cumulativeLower[i] = lower.GetSurfaceArea();        
			cumulativeUpper[numFaces-i-1] = upper.GetSurfaceArea();
		}

		float invTotalSA = 1.0f / cumulativeUpper[0];

		// test all split positions
		for (uint32 i=0; i < numFaces-1; ++i)
		{
			float pBelow = cumulativeLower[i] * invTotalSA;
			float pAbove = cumulativeUpper[i] * invTotalSA;

			float cost = 0.125f + (pBelow*i + pAbove*(numFaces-i));
			if (cost <= bestCost)
			{
				bestCost = cost;
				bestIndex = i;
				bestAxis = a;
			}
		}
	}

	// re-sort by best axis
	FaceSorter predicate(&m_vertices[0], &m_indices[0], m_numFaces*3, bestAxis);
	std::sort(faces, faces+numFaces, predicate);

	return bestIndex+1;
}

void AABBTree::BuildRecursive(uint32 nodeIndex, uint32* faces, uint32 numFaces)
{
    const uint32 kMaxFacesPerLeaf = 6;
    
    // if we've run out of nodes allocate some more
    if (nodeIndex >= m_nodes.size())
    {
		uint32 s = std::max(uint32(1.5f*m_nodes.size()), 512U);

		cout << "Resizing tree, current size: " << m_nodes.size()*sizeof(Node) << " new size: " << s*sizeof(Node) << endl;

        m_nodes.resize(s);
    }

    // a reference to the current node, need to be careful here as this reference may become invalid if array is resized
	Node& n = m_nodes[nodeIndex];

	// track max tree depth
    ++s_depth;
    m_treeDepth = max(m_treeDepth, s_depth);

	CalculateFaceBounds(faces, numFaces, n.m_minExtents, n.m_maxExtents);

	// calculate bounds of faces and add node  
    if (numFaces <= kMaxFacesPerLeaf)
    {
        n.m_faces = faces;
        n.m_numFaces = numFaces;		

        ++m_leafNodes;
    }
    else
    {
        ++m_innerNodes;        

        // face counts for each branch
        //const uint32 leftCount = PartitionMedian(n, faces, numFaces);
        const uint32 leftCount = PartitionSAH(n, faces, numFaces);
        const uint32 rightCount = numFaces-leftCount;

		// alloc 2 nodes
		m_nodes[nodeIndex].m_children = m_freeNode;

		// allocate two nodes
		m_freeNode += 2;
  
        // split faces in half and build each side recursively
        BuildRecursive(m_nodes[nodeIndex].m_children+0, faces, leftCount);
        BuildRecursive(m_nodes[nodeIndex].m_children+1, faces+leftCount, rightCount);
    }

    --s_depth;
}

struct StackEntry
{
    uint32 m_node;   
    float m_dist;
};


#define TRACE_STATS 0


bool AABBTree::TraceRay(const Point3& start, const Vector3& dir, float& outT, float& outU, float& outV, float& outW, float& outSign, uint32& outIndex) const
{
#if _WIN32
    // reset stats
    s_traceDepth = 0;
#endif
	
    const Vector3 rcp_dir(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);

	// some temp variables
	Vector3 normal;
    float t, u, v, w, s;

    float minT = FLT_MAX;
    float minU, minV, minW, minSign;
    uint32 minIndex;
    Vector3 minNormal;

    const uint32 kStackDepth = 50;
    
    StackEntry stack[kStackDepth];
    stack[0].m_node = 0;
    stack[0].m_dist = 0.0f;

    uint32 stackCount = 1;

    while (stackCount)
    {
        // pop node from back
        StackEntry& e = stack[--stackCount];
        
        // ignore if another node has already come closer
        if (e.m_dist >= minT)
        {
            continue;
        }

        const Node* node = &m_nodes[e.m_node];

filth:

        if (node->m_faces == NULL)
        {

#if TRACE_STATS
            extern uint32 g_nodesChecked;
            ++g_nodesChecked;
#endif

#if _WIN32
			++s_traceDepth;
#endif
            // find closest node
            const Node& leftChild = m_nodes[node->m_children+0];
            const Node& rightChild = m_nodes[node->m_children+1];

            float dist[2] = {FLT_MAX, FLT_MAX};

            IntersectRayAABBOmpf(start, rcp_dir, leftChild.m_minExtents, leftChild.m_maxExtents, dist[0]);
            IntersectRayAABBOmpf(start, rcp_dir, rightChild.m_minExtents, rightChild.m_maxExtents, dist[1]);

            const uint32 closest = dist[1] < dist[0]; // 0 or 1
            const uint32 furthest = closest ^ 1;

            if (dist[furthest] < minT)
            {
                StackEntry& e = stack[stackCount++];
                e.m_node = node->m_children+furthest;
                e.m_dist = dist[furthest];
            }

            // lifo
            if (dist[closest] < minT)
            {
                node = &m_nodes[node->m_children+closest];
                goto filth;
            }
        }
        else
        {
            for (uint32 i=0; i < node->m_numFaces; ++i)
            {
                const uint32 faceIndex = node->m_faces[i];
                const uint32 indexStart = faceIndex*3;

                const Point3& a = m_vertices[m_indices[indexStart+0]];
                const Point3& b = m_vertices[m_indices[indexStart+1]];
                const Point3& c = m_vertices[m_indices[indexStart+2]];
#if TRACE_STATS
                extern uint32 g_trisChecked;
                ++g_trisChecked;
#endif

                if (IntersectRayTriTwoSided(start, dir, a, b, c, t, u, v, w, s))
                {
                    if (t < minT && t > 0.01f)
                    {
                        minT = t;
                        minU = u;
                        minV = v;
                        minW = w;
						minSign = s;
                        minIndex = faceIndex;
                    }
                }
            }
        }
    }

    // copy to outputs
    outT = minT;
    outU = minU;
    outV = minV;
    outW = minW;
	outSign = minSign;
    outIndex = minIndex;

    return (outT != FLT_MAX);
}
/*
bool AABBTree::TraceRay(const Point3& start, const Vector3& dir, float& outT, float& u, float& v, float& w, float& faceSign, uint32& faceIndex) const
{   
    s_traceDepth = 0;

    Vector3 rcp_dir(1.0f/dir.x, 1.0f/dir.y, 1.0f/dir.z);

    outT = FLT_MAX;
    TraceRecursive(0, start, dir, outT, u, v, w, faceSign, faceIndex);

    return (outT != FLT_MAX);
}
*/

void AABBTree::TraceRecursive(uint32 nodeIndex, const Point3& start, const Vector3& dir, float& outT, float& outU, float& outV, float& outW, float& faceSign, uint32& faceIndex) const
{
	const Node& node = m_nodes[nodeIndex];

    if (node.m_faces == NULL)
    {
#if _WIN32
        ++s_traceDepth;
#endif
		
#if TRACE_STATS
        extern uint32 g_nodesChecked;
        ++g_nodesChecked;
#endif

        // find closest node
        const Node& leftChild = m_nodes[node.m_children+0];
        const Node& rightChild = m_nodes[node.m_children+1];

        float dist[2] = {FLT_MAX, FLT_MAX};

        IntersectRayAABB(start, dir, leftChild.m_minExtents, leftChild.m_maxExtents, dist[0], NULL);
        IntersectRayAABB(start, dir, rightChild.m_minExtents, rightChild.m_maxExtents, dist[1], NULL);
        
        uint32 closest = 0;
        uint32 furthest = 1;
		
        if (dist[1] < dist[0])
        {
            closest = 1;
            furthest = 0;
        }		

        if (dist[closest] < outT)
            TraceRecursive(node.m_children+closest, start, dir, outT, outU, outV, outW, faceSign, faceIndex);

        if (dist[furthest] < outT)
            TraceRecursive(node.m_children+furthest, start, dir, outT, outU, outV, outW, faceSign, faceIndex);

    }
    else
    {
        Vector3 normal;
        float t, u, v, w, s;

        for (uint32 i=0; i < node.m_numFaces; ++i)
        {
            uint32 indexStart = node.m_faces[i]*3;

            const Point3& a = m_vertices[m_indices[indexStart+0]];
            const Point3& b = m_vertices[m_indices[indexStart+1]];
            const Point3& c = m_vertices[m_indices[indexStart+2]];
#if TRACE_STATS
            extern uint32 g_trisChecked;
            ++g_trisChecked;
#endif

            if (IntersectRayTriTwoSided(start, dir, a, b, c, t, u, v, w, s))
            {
                if (t < outT)
                {
                    outT = t;
					outU = u;
					outV = v;
					outW = w;
					faceSign = s;
					faceIndex = node.m_faces[i];
                }
            }
        }
    }
}

/*
bool AABBTree::TraceRay(const Point3& start, const Vector3& dir, float& outT, Vector3* outNormal) const
{   
    outT = FLT_MAX;
    TraceRecursive(0, start, dir, outT, outNormal);

    return (outT != FLT_MAX);
}

void AABBTree::TraceRecursive(uint32 n, const Point3& start, const Vector3& dir, float& outT, Vector3* outNormal) const
{
    const Node& node = m_nodes[n];

    if (node.m_numFaces == 0)
    {
        extern _declspec(thread) uint32 g_traceDepth;
        ++g_traceDepth;
#if _DEBUG
        extern uint32 g_nodesChecked;
        ++g_nodesChecked;
#endif
        float t;
        if (IntersectRayAABB(start, dir, node.m_minExtents, node.m_maxExtents, t, NULL))
        {
            if (t <= outT)
            {
                TraceRecursive(n*2+1, start, dir, outT, outNormal);
                TraceRecursive(n*2+2, start, dir, outT, outNormal);              
            }
        }
    }
    else
    {
        Vector3 normal;
        float t, u, v, w;

        for (uint32 i=0; i < node.m_numFaces; ++i)
        {
            uint32 indexStart = node.m_faces[i]*3;

            const Point3& a = m_vertices[m_indices[indexStart+0]];
            const Point3& b = m_vertices[m_indices[indexStart+1]];
            const Point3& c = m_vertices[m_indices[indexStart+2]];
#if _DEBUG
            extern uint32 g_trisChecked;
            ++g_trisChecked;
#endif

            if (IntersectRayTri(start, dir, a, b, c, t, u, v, w, &normal))
            {
                if (t < outT)
                {
                    outT = t;

                    if (outNormal)
                        *outNormal = normal;
                }
            }
        }
    }
}
*/
bool AABBTree::TraceRaySlow(const Point3& start, const Vector3& dir, float& outT, Vector3* outNormal) const
{    
    const uint32 numFaces = GetNumFaces();

    float minT = FLT_MAX;
    Vector3 minNormal(0.0f, 1.0f, 0.0f);

    Vector3 n(0.0f, 1.0f, 0.0f);
    float t = 0.0f;
    bool hit = false;

    for (uint32 i=0; i < numFaces; ++i)
    {
        const Point3& a = m_vertices[m_indices[i*3+0]];
        const Point3& b = m_vertices[m_indices[i*3+1]];
        const Point3& c = m_vertices[m_indices[i*3+2]];

        float u, v, w;
        if (IntersectRayTri(start, dir, a, b, c, t, u, v, w, &n))
        {
            if (t < minT)
            {
                minT = t;
                minNormal = n;
                hit = true;
            }
        }
    }

    outT = t;
    if (outNormal)
    {
        *outNormal = Normalize(minNormal);
    }

    return hit;
}

void AABBTree::DebugDraw()
{
	/*
    glPolygonMode( GL_FRONT_AND_BACK, GL_LINE );

    DebugDrawRecursive(0, 0);

    glPolygonMode( GL_FRONT_AND_BACK, GL_FILL );
	*/

}

void AABBTree::DebugDrawRecursive(uint32 nodeIndex, uint32 depth)
{
    static uint32 kMaxDepth = 3;

    if (depth > kMaxDepth)
        return;


    /*
    Node& n = m_nodes[nodeIndex];

	Vector3 minExtents = FLT_MAX;
    Vector3 maxExtents = -FLT_MAX;

    // calculate face bounds
    for (uint32 i=0; i < m_vertices.size(); ++i)
    {
        Vector3 a = m_vertices[i];

        minExtents = Min(a, minExtents);
        maxExtents = Max(a, maxExtents);
    }

    
    glBegin(GL_QUADS);
    glVertex3f(minExtents.x, maxExtents.y, 0.0f);
    glVertex3f(maxExtents.x, maxExtents.y, 0.0f);
    glVertex3f(maxExtents.x, minExtents.y, 0.0f);
    glVertex3f(minExtents.x, minExtents.y, 0.0f);
    glEnd();
    

    n.m_center = Point3(minExtents+maxExtents)/2;
    n.m_extents = (maxExtents-minExtents)/2;
    */
/*
    if (n.m_minEtextents != Vector3(0.0f))
    {
        Point3 corners[8];
        corners[0] = n.m_center + Vector3(-n.m_extents.x, n.m_extents.y, n.m_extents.z);
        corners[1] = n.m_center + Vector3(n.m_extents.x, n.m_extents.y, n.m_extents.z);
        corners[2] = n.m_center + Vector3(n.m_extents.x, -n.m_extents.y, n.m_extents.z);
        corners[3] = n.m_center + Vector3(-n.m_extents.x, -n.m_extents.y, n.m_extents.z);

        corners[4] = n.m_center + Vector3(-n.m_extents.x, n.m_extents.y, -n.m_extents.z);
        corners[5] = n.m_center + Vector3(n.m_extents.x, n.m_extents.y, -n.m_extents.z);
        corners[6] = n.m_center + Vector3(n.m_extents.x, -n.m_extents.y, -n.m_extents.z);
        corners[7] = n.m_center + Vector3(-n.m_extents.x, -n.m_extents.y, -n.m_extents.z);
        
        glBegin(GL_QUADS);
        glColor3f(0.0f, 1.0f, 0.0f);
        glVertex3fv(corners[0]);
        glVertex3fv(corners[1]);
        glVertex3fv(corners[2]);
        glVertex3fv(corners[3]);

        glVertex3fv(corners[1]);
        glVertex3fv(corners[5]);
        glVertex3fv(corners[6]);
        glVertex3fv(corners[2]);

        glVertex3fv(corners[0]);
        glVertex3fv(corners[4]);
        glVertex3fv(corners[5]);
        glVertex3fv(corners[1]);

        glVertex3fv(corners[4]);
        glVertex3fv(corners[5]);
        glVertex3fv(corners[6]);
        glVertex3fv(corners[7]);

        glVertex3fv(corners[0]);
        glVertex3fv(corners[4]);
        glVertex3fv(corners[7]);
        glVertex3fv(corners[3]);

        glVertex3fv(corners[3]);
        glVertex3fv(corners[7]);
        glVertex3fv(corners[6]);
        glVertex3fv(corners[2]);

        glEnd();            

        DebugDrawRecursive(nodeIndex*2+1, depth+1);
        DebugDrawRecursive(nodeIndex*2+2, depth+1);
    }    
    */
}
