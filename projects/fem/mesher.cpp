#include "mesher.h"

#include "core/maths.h"

#include <vector>

using namespace std;

#define AssertEq(x, y, e) { if (Abs(x-y) > e*Max(Abs(x), Abs(y))) { cout << #x << " != " << #y << "(" << x - y << ")" << endl; assert(0); } } 

namespace
{

	bool CalculateCircumcircle(const Vec2& p, const Vec2& q, const Vec2& r, Vec2& outCircumCenter, float& outCircumRadius)
	{
		printf("(%f, %f) (%f, %f) (%f, %f)\n", p.x, p.y, q.x, q.y, r.x, r.y);

		// calculate the intersection of two perpendicular bisectors
		const Vec2 pq = q-p;
		const Vec2 qr = r-q;

		// check winding
		printf("%f\n", Cross(pq, qr));
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
		outCircumRadius = Length(outCircumCenter-p);
		
		//printf("(%f, %f) (%f, %f) (%f, %f):  %f, %f: %f\n", p.x, p.y, q.x, q.y, r.x, r.y, outCircumCenter.x, outCircumCenter.y, outCircumRadiusSq);

		// sanity check center is equidistant from other vertices
		AssertEq(Length(outCircumCenter-q), outCircumRadius, 0.01f);
		AssertEq(Length(outCircumCenter-r), outCircumRadius, 0.01f);	

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

			CalculateCircumcircle(p[i], p[j], p[k], mCircumCenter, mCircumRadius);
		}

		Edge mEdges[3];

		Vec2 mCircumCenter;
		float mCircumRadius;
	};

	float TriArea(const Vec2& a, const Vec2& b, const Vec2& c)
	{
		return 0.5f*(Cross(b-a, c-a)); 
	}

	float TriMinAngle(const Vec2& a, const Vec2& b, const Vec2& c)
	{
		Vec2 e1 = Normalize(b-a);
		Vec2 e2 = Normalize(c-a);
	   	Vec2 e3 = Normalize(c-b);

		float alpha = acosf(Dot(e1, e2));
		float beta = acosf(Dot(-e1, e3));
		float gamma = kPi-alpha-beta;

		return min(min(alpha, beta), gamma);
	}

	bool PointInTri(const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& p)
	{
		assert(TriArea(a, b, c) > 0.0f);

		// compute signed area of subtris, assumes tris wound CCW
		bool outside = TriArea(p, a, b) < 0.0f || TriArea(p, b, c) < 0.0f || TriArea(p, c, a) < 0.0f;

		return !outside;
	}
/*
	struct Triangulation
	{
		vector<Vec2> vertices;
		vector<Triangle> triangles;

		Triangulation(const Vec2& lower, const Vec2& upper)
		{
			// initialize triangulation with the bounding box
			vertices.push_back(Vec2(lower.x, upper.x));
			vertices.push_back(Vec2(lower.x, lower.y));
			vertices.push_back(Vec2(upper.x, lower.y));
			vertices.push_back(Vec2(upper.x, upper.y));

			triangles.push_back(Triangle(0, 1, 2, &vertices[0]);
			triangles.push_back(Triangle(2, 1, 3, &vertices[0]);
		}

		void Insert(const Vec2& p)
		{
		}
	};
*/
};

// incremental insert Delaunay triangulation based on Bowyer/Watson's algorithm
//
void TriangulateDelaunay(const Vec2* points, uint32_t numPoints, uint32_t maxPoints, float maxArea, float minAngle, vector<uint32_t>& outTris, vector<Vec2>& outPoints)
{

/* 
	Vec2 t1[3] = { Vec2(0.090023, 0.865998), Vec2(0.053043, 0.581536), Vec2(0.316703, 1.078273) };
	Vec2 t2[3] = { Vec2(0.720000, 0.160000), Vec2(0.720000, 0.080000), Vec2(1.200000, 0.160000) };

	float r1, r2;
	Vec2 c1, c2;
   
	CalculateCircumcircle(t1[0], t1[1], t1[2], c1, r1);
	CalculateCircumcircle(t2[0], t2[1], t2[2], c2, r2);

	printf("%f %f: %f\n", c1.x, c1.y, r1);
	printf("%f %f: %f\n", c2.x, c2.y, r2);

	exit(0);
*/
	vector<Vec2> vertices(points, points+numPoints);
	vector<Triangle> triangles;
	vector<Edge> edges;

	// alloc room for most points
	vertices.resize(maxPoints);

	// initialize with an all containing triangle, todo: calculate proper bounds
	const float bounds = 20.0f;
	vertices.push_back(Vec2(-bounds, -bounds));
	vertices.push_back(Vec2( bounds, -bounds));
	vertices.push_back(Vec2( 0.0f,  bounds));

	Triangle seed(maxPoints, maxPoints+1, maxPoints+2, &vertices[0]);
	triangles.push_back(seed);

	uint32_t i=0;

	// insert / refine loop
	while (i < numPoints)
	{
		// insert each vertex
		for (; i < numPoints; ++i)
		{
			edges.resize(0);

			const Vec2 p = vertices[i];

			//printf("Point: %f %f\n", points[i].x, points[i].y);

			printf("tris = {\n");
			// find all triangles for which inserting this point would
			// violate the Delaunay condition, that is, which triangles
			// circumcircles does this point lie inside
			for (uint32_t j=0; j < triangles.size(); )
			{
				const Triangle& t = triangles[j];

				if (Length(t.mCircumCenter-p) <= t.mCircumRadius)
				{
					//printf("%d %d %d by %f, %f\n", t.mEdges[0][0], t.mEdges[0][1], t.mEdges[1][1], LengthSq(t.mCircumCenter-p), t.mCircumRadiusSq);

					Vec2 a = vertices[t.mEdges[0][0]];
					Vec2 b = vertices[t.mEdges[0][1]];
					Vec2 c = vertices[t.mEdges[1][1]];

					/*
					printf("	Inside tri: (%f, %f) (%f, %f) (%f, %f) with %f, %f radius: %f dist: %f\n",
						   	a.x, a.y, b.x, b.y, c.x, c.y, t.mCircumCenter.x, t.mCircumCenter.y, t.mCircumRadius,
						   	LengthSq(t.mCircumCenter-p));
					*/
					printf("{{%f, %f}, {%f, %f}, {%f, %f}},\n",a.x, a.y, b.x, b.y, c.x, c.y); 

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

			printf("}\npr = {%f, %f}\nedges = {", p.x, p.y);

			for (uint32_t e=0; e < edges.size(); ++e)
			{
				printf("{{%f, %f}, {%f, %f}},\n",
					vertices[edges[e][0]].x, vertices[edges[e][0]].y,
					vertices[edges[e][1]].x, vertices[edges[e][1]].y);
			}
		
			printf("}\n");

			// re-triangulate point to the enclosing set of edges
			for (uint32_t e=0; e < edges.size(); ++e)
			{
				Triangle t(edges[e][0], edges[e][1], i, &vertices[0]);
				triangles.push_back(t);
			}
		}	

		
		printf("----------------------------------------------------\nrefine pass\n");	
		
		// perform a refinement stage, inserting vertices at the
		// circumcenter of any triangles that don't fit our quality criteria
		for (uint32_t r=0; r < triangles.size() && numPoints < maxPoints; ++r)
		{
			const Triangle& t = triangles[r];

			uint32_t i = t.mEdges[0][0];
			uint32_t j = t.mEdges[0][1];
			uint32_t k = t.mEdges[1][1];
			
			Vec2 a = vertices[t.mEdges[0][0]];
			Vec2 b = vertices[t.mEdges[0][1]];
			Vec2 c = vertices[t.mEdges[1][1]];

			if (i < maxPoints && j < maxPoints && k < maxPoints)
			{
				if (fabsf(TriArea(a,b,c)) > maxArea || TriMinAngle(a,b,c) < minAngle)
				{
					// insert new vertices
					printf("refining tri %d with indices: (%d, %d, %d) at: %f %f area: %f minangle: %f\n", r, i, j, k, t.mCircumCenter.x, t.mCircumCenter.y, TriArea(a,b,c), TriMinAngle(a,b,c));
					printf("and vertices (%f, %f), (%f, %f), (%f, %f)\n", a.x, a.y, b.x, b.y, c.x, c.y);

					bool insideHull = false;

					Vec2 p = t.mCircumCenter;

					// check if circumcenter lies inside the mesh
					for (uint32_t s=0; s < triangles.size(); ++s)
					{
						const Triangle& t = triangles[s];

						Vec2 a = vertices[t.mEdges[0][0]];
						Vec2 b = vertices[t.mEdges[0][1]];
						Vec2 c = vertices[t.mEdges[1][1]];
			
						if (t.mEdges[0][0] < maxPoints && t.mEdges[0][1] < maxPoints && t.mEdges[1][1] < maxPoints && PointInTri(a, b, c, p))
						{
							printf("pt inside: {{%f, %f}, {%f, %f}, {%f, %f}} - {%f, %f}\n", a.x, a.y, b.x, b.y, c.x, c.y, t.mCircumCenter.x, t.mCircumCenter.y);
							insideHull = true;
							break;
						}
					}	
					// one refine at a time
					if (insideHull)
					{
						vertices[numPoints++] = p;
						break;
					}
				}
			}
		}
	}

	// copy to output
	outPoints.assign(vertices.begin(), vertices.begin()+numPoints);

	outTris.reserve(triangles.size()*3);

	for (uint32_t i=0; i < triangles.size(); ++i)
	{
		const Triangle& t = triangles[i];
		const uint32_t v[3] = { t.mEdges[0][0], t.mEdges[0][1], t.mEdges[1][1] };

		// throw away tris connected to the initial enclosing tri	
		if (v[0] < maxPoints && v[1] < maxPoints && v[2] < maxPoints)
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
