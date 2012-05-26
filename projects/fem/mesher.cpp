#include "mesher.h"

#include "core/maths.h"

#include <vector>

using namespace std;

#define AssertEq(x, y, e) { if (Abs(x-y) > e*Max(Abs(x), Abs(y))) { cout << #x << " != " << #y << "(" << x - y << ")" << endl; assert(0); } } 

namespace
{

	bool CalculateCircumcircle(const Vec2& p, const Vec2& q, const Vec2& r, Vec2& outCircumCenter, float& outCircumRadiusSq)
	{
		// calculate the intersection of two perpendicular bisectors
		const Vec2 pq = q-p;
		const Vec2 qr = r-q;

		// check winding
		assert(Cross(pq, qr) >= 0.0f); 

		// mid-points of  edges 
		const Vec2 a = 0.5f*(p+q);
		const Vec2 b = 0.5f*(q+r);
		const Vec2 u = PerpCCW(pq);

		const float d = Dot(u, qr);

		// check if degenerate
		assert(d != 0.0f);

		const float t = Dot(b-a, qr)/d;
		
		outCircumCenter = a + t*u;
		outCircumRadiusSq = LengthSq(outCircumCenter-p);

		//printf("(%f, %f) (%f, %f) (%f, %f):  %f, %f: %f\n", p.x, p.y, q.x, q.y, r.x, r.y, outCircumCenter.x, outCircumCenter.y, outCircumRadiusSq);

		// sanity check center is equidistant from other vertices
		AssertEq(LengthSq(outCircumCenter-q), outCircumRadiusSq, 0.01f);
		AssertEq(LengthSq(outCircumCenter-r), outCircumRadiusSq, 0.01f);	

		return true;
	}

	struct Edge
	{
		Edge() {};
		Edge(uint32_t i, uint32_t j)
		{
			mIndices[0] = i;
			mIndices[1] = j;
		}

		bool operator==(const Edge& e) const
		{
			return ((*this)[0] == e[0] && (*this)[1] == e[1]) ||
				   ((*this)[0] == e[1] && (*this)[1] == e[0]);
		}

		uint32_t operator[](uint32_t index) const
		{
			assert(index < 2);
			return mIndices[index];
		}

		uint32_t mIndices[2];
	};

	struct Triangle
	{
		Triangle(uint32_t i, uint32_t j, uint32_t k, const Vec2* p)
		{
			mEdges[0] = Edge(i, j);
			mEdges[1] = Edge(j, k);
			mEdges[2] = Edge(k, i);

			CalculateCircumcircle(p[i], p[j], p[k], mCircumCenter, mCircumRadiusSq);
		}

		Edge mEdges[3];

		Vec2 mCircumCenter;
		float mCircumRadiusSq;
	};
};
// incremental insert Delaunay triangulation
void TriangulateDelaunay(const Vec2* points, uint32_t numPoints, vector<uint32_t>& outTris)
{
	vector<Vec2> vertices(points, points+numPoints);
	vector<Triangle> triangles;
	vector<Edge> edges;

	// initialize with an all containing triangle, todo: calculate proper bounds
	const float bounds = 1000.0f;

	vertices.push_back(Vec2(-bounds, -bounds));
	vertices.push_back(Vec2( bounds, -bounds));
	vertices.push_back(Vec2( 0.0f,  bounds));

	Triangle seed(numPoints, numPoints+1, numPoints+2, &vertices[0]);
	triangles.push_back(seed);
	
	for (uint32_t i=0; i < numPoints; ++i)
	{
		edges.resize(0);

		const Vec2 p = points[i];

		// find all triangles for which inserting this point would
		// violate the Delaunay condition, that is, which triangles
		// circumcircles does this point lie inside
		for (uint32_t j=0; j < triangles.size(); )
		{
			const Triangle& t = triangles[j];
			
			if (LengthSq(t.mCircumCenter-p) < t.mCircumRadiusSq)
			{
				for (uint32_t e=0; e < 3; ++e)
				{
					// if edge doesn't already exist add it
					vector<Edge>::iterator it = find(edges.begin(), edges.end(), t.mEdges[e]);

					if (it == edges.end())
						edges.push_back(t.mEdges[e]);
					else
						edges.erase(it);
				}	

				// remove triangle
				triangles.erase(triangles.begin()+j);
			}
			else
			{
				// next triangle
				++j;
			}
		}	

		// re-triangulate point to the enclosing set of edges
		for (uint32_t e=0; e < edges.size(); ++e)
		{
			Triangle t(edges[e][0], edges[e][1], i, &vertices[0]);
			triangles.push_back(t);
		}
	}	

	// copy to output
	outTris.reserve(triangles.size()*3);

	for (uint32_t i=0; i < triangles.size(); ++i)
	{
		const Triangle& t = triangles[i];

		const uint32_t v[3] = { t.mEdges[0][0], t.mEdges[0][1], t.mEdges[1][1] };
	
		if (v[0] < numPoints && v[1] < numPoints && v[2] < numPoints)
			outTris.insert(outTris.end(), v, v+3);
	}
}

void CreateTorus(std::vector<Vec2>& points, std::vector<uint32_t>& indices, float inner, float outer, uint32_t segments)
{
	assert(inner < outer);

	for (uint32_t i=0; i < segments; ++i)
	{
		float theta = float(i)/segments*kPi*2.0f;
		
		float x = sinf(theta);
		float y = cosf(theta);
		
		points.push_back(Vec2(x, y)*outer);
		points.push_back(Vec2(x, y)*inner);

		if (i > 0)
		{
			uint32_t base = (i-1)*2;

			indices.push_back(base+0);
			indices.push_back(base+1);
			indices.push_back(base+2);

			indices.push_back(base+2);
			indices.push_back(base+1);
			indices.push_back(base+3);
		}
	}

	uint32_t base = points.size()-2;

	indices.push_back(base+0);
	indices.push_back(base+1);
	indices.push_back(0);

	indices.push_back(0);
	indices.push_back(base+1);
	indices.push_back(1);
}
