#include "mesher.h"

#include "core/maths.h"

#include <vector>

using namespace std;

#define DEBUG_PRINT 0

typedef float real;
typedef XVector2<real> Vec2r;

namespace
{
	bool CalculateCircumcircle(const Vec2r& p, const Vec2r& q, const Vec2r& r, Vec2r& outCircumCenter, real& outCircumRadius)
	{
#if DEBUG_PRINT
		printf("re = {{%f, %f}, {%f, %f}, {%f, %f}}\n", p.x, p.y, q.x, q.y, r.x, r.y);
#endif
		// calculate the intersection of two perpendicular bisectors
		const Vec2r pq = q-p;
		const Vec2r qr = r-q;

		// check winding
#if DEBUG_PRINT
		printf("%f\n", Cross(pq, qr));
#endif
		assert(Cross(pq, qr) >= 0.0f); 

		// mid-points of  edges 
		const Vec2r a = real(0.5)*(p+q);
		const Vec2r b = real(0.5)*(q+r);
		const Vec2r u = PerpCCW(pq);

		const real d = Dot(u, qr);

		// check if degenerate
		assert(d != 0.0f);

		const real t = Dot(b-a, qr)/d;
		
		outCircumCenter = a + t*u;
		outCircumRadius = Length(outCircumCenter-p);
		
		return true;
	}

	struct Edge
	{
		Edge() {};
		Edge(uint32_t i, uint32_t j)
		{
			mIndices[0] = i;
			mIndices[1] = j;

			mFaces[0] = uint32_t(-1);
		    mFaces[1] = uint32_t(-1);	
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
		uint32_t mFaces[2];
	};

	struct Triangle
	{
		Triangle(uint32_t i, uint32_t j, uint32_t k, const Vec2r* p)
		{
			mVertices[0] = i;
			mVertices[1] = j; 
			mVertices[2] = k; 

			CalculateCircumcircle(p[i], p[j], p[k], mCircumCenter, mCircumRadius);
		}

		uint32_t mVertices[3];

		Vec2r mCircumCenter;
		real mCircumRadius;
	};

	real TriArea(const Vec2r& a, const Vec2r& b, const Vec2r& c)
	{
		return 0.5f*(Cross(b-a, c-a)); 
	}

	real TriMinAngle(const Vec2r& a, const Vec2r& b, const Vec2r& c)
	{
		Vec2r e1 = Normalize(b-a);
		Vec2r e2 = Normalize(c-a);
	   	Vec2r e3 = Normalize(c-b);

		real alpha(acos(Dot(e1, e2)));
		real beta(acos(Dot(-e1, e3)));
		real gamma(kPi-alpha-beta);

		return min(min(alpha, beta), gamma);
	}

	bool PointInTri(const Vec2r& a, const Vec2r& b, const Vec2r& c, const Vec2r& p)
	{
		assert(TriArea(a, b, c) > 0.0f);

		// compute signed area of subtris, assumes tris wound CCW
		bool outside = TriArea(p, a, b) < 0.0f || TriArea(p, b, c) < 0.0f || TriArea(p, c, a) < 0.0f;

		return !outside;
	}

	struct Triangulation
	{
		vector<Vec2r> vertices;
		vector<Triangle> triangles;
		vector<Edge> segments;

		Triangulation(const Vec2* points, uint32_t numPoints, const uint32_t* tris, uint32_t numTris) 
		{
			for (uint32_t i=0; i < numPoints; ++i)
				vertices.push_back(Vec2r(points[i]));

			for (uint32_t i=0; i < numTris; ++i)
				triangles.push_back(Triangle(tris[i*3+0], tris[i*3+1], tris[i*3+2], &vertices[0]));

			assert(Valid());
		}

		Triangulation(const Vec2r& lower, const Vec2r& upper) 
		{
			Vec2r extents(upper-lower);

			// initialize triangulation with the bounding box
			vertices.push_back(lower);
			vertices.push_back(lower + real(2.0)*Vec2r(extents.x, real(0.0)));	
			vertices.push_back(lower + real(2.0)*Vec2r(real(0.0), extents.y));

			triangles.push_back(Triangle(0, 1, 2, &vertices[0]));

			assert(Valid());
		}

		void Insert(Vec2r p)
		{
			vector<Edge> edges;

			uint32_t i = vertices.size();
			vertices.push_back(p);

#if DEBUG_PRINT
			printf("tris = {\n");
#endif

			// find all triangles for which inserting this point would
			// violate the Delaunay condition, that is, which triangles
			// circumcircles does this point lie inside
			for (uint32_t j=0; j < triangles.size(); )
			{
				const Triangle& t = triangles[j];

					Vec2r a = vertices[t.mVertices[0]];
					Vec2r b = vertices[t.mVertices[1]];
					Vec2r c = vertices[t.mVertices[2]];
#if DEBUG_PRINT
					printf("{{%f, %f}, {%f, %f}, {%f, %f}},\n",a.x, a.y, b.x, b.y, c.x, c.y); 
#endif

				if (Length(t.mCircumCenter-p) <= t.mCircumRadius)
				{
					for (uint32_t e=0; e < 3; ++e)
					{
						Edge edge(t.mVertices[e], t.mVertices[(e+1)%3]);

						// if edge doesn't already exist add it
						vector<Edge>::iterator it = find(edges.begin(), edges.end(), edge); 

						if (it == edges.end())
							edges.push_back(edge);
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

#if DEBUG_PRINT
			printf("}\npr = {%f, %f}\nedges = {", p.x, p.y);

			for (uint32_t e=0; e < edges.size(); ++e)
			{
				printf("{{%f, %f}, {%f, %f}},\n",
					vertices[edges[e][0]].x, vertices[edges[e][0]].y,
				vertices[edges[e][1]].x, vertices[edges[e][1]].y);
			}
		
			printf("}\n");
#endif
			// re-triangulate point to the enclosing set of edges
			for (uint32_t e=0; e < edges.size(); ++e)
			{
				
				uint32_t v0 = edges[e][0];
				uint32_t v1 = edges[e][1];
				uint32_t v2 = i;

				if (fabs(TriArea(vertices[v0], vertices[v1], vertices[v2])) > real(1.e-5))
				{
					Triangle t(edges[e][0], edges[e][1], i, &vertices[0]);
					triangles.push_back(t);
				}
			}

			//assert(Valid());
		}

		bool ContainsPoint(const Vec2r& c) const
		{
			for (uint32_t i=0; i < triangles.size(); ++i)
			{
				const Triangle& t = triangles[i];

				// ignore border triangles
				if (t.mVertices[0] < 3 || t.mVertices[1] < 3 || t.mVertices[2] < 3)
					continue;

				Vec2r u = vertices[t.mVertices[0]];
				Vec2r v = vertices[t.mVertices[1]];
				Vec2r w = vertices[t.mVertices[2]];
				
				if (PointInTri(u, v, w, c))
					return true;
			}

			return false;
		}

		// finds an encroached segment and return it's diametral circle
		bool FindEncroachedSegment(Vec2r& c, real& r) const
		{
			// try and find an enchroached segment
			for (uint32_t i=0; i < triangles.size(); ++i)
			{
				const Triangle& t = triangles[i];

				for (uint32_t e=0; e < 3; ++e)
				{
					uint32_t i0 = t.mVertices[e];
					uint32_t i1 = t.mVertices[(e+1)%3];

					// ignore border edges 
					//if (i0 < 3 || i1 < 3)
				//		continue;

					Vec2r p = vertices[i0];
					Vec2r q = vertices[i1];

					// calculate edge midpoint and radius
					Vec2r midpoint = (p+q)*0.5f;
					real radius = 0.5f*Length(p-q);

					// see check if edge is encroached
					for (uint32_t v=0; v < vertices.size(); ++v)
					{
						if (Length(vertices[v]-midpoint) < radius && v != i0 && v != i1)
						{
							c = midpoint;
							r = radius;	
							return true;
						}
					}	
				}	
			}

			return false;
		}

		int FindPoorQualityTriangle(real minAngle, real maxArea)
		{
			const real b = real(1.0)/(real(2.0f)*sin(minAngle));
			real maxB = b;
			int worst = -1; 

			for (uint32_t i=0; i < triangles.size(); ++i)
			{
				const Triangle& t = triangles[i];

				Vec2r c = t.mCircumCenter;
		
				//if (!TestVisiblity(c, t))
				if (!ContainsPoint(c))
					continue;
					
				// calculate ratio of circumradius to shortest edge
				real minEdgeLength = FLT_MAX;

				for (uint32_t e=0; e < 3; ++e)
				{
					Vec2r p = vertices[t.mVertices[e]];
					Vec2r q = vertices[t.mVertices[(e+1)%3]];

					minEdgeLength = min(minEdgeLength, Length(p-q));
				}

				real r = t.mCircumRadius / minEdgeLength;
				
				if (r > maxB)
				{
					maxB = r;
					worst = i;
					return i;
				}
	
				
				Vec2r u= vertices[t.mVertices[0]];
				Vec2r v = vertices[t.mVertices[1]];
				Vec2r w = vertices[t.mVertices[2]];
				
				if (TriArea(u, v, w) > maxArea)
					return i;				
					
			}
		
			return worst;
		}

		bool Valid()
		{
			for (uint32_t i=0; i < triangles.size(); ++i)
			{
				const Triangle& t = triangles[i];

				for (uint32_t j=0; j < vertices.size(); ++j)
				{
					if (t.mVertices[0] == j ||
						t.mVertices[1] == j ||
						t.mVertices[2] == j)
						continue;

					real eps = 1.e-4f;
					real d = Length(t.mCircumCenter-vertices[j]);

					if (d < t.mCircumRadius-eps)
						return false;
				}
			}

			return true;
		}
	};

};

// incremental insert Delaunay triangulation based on Bowyer/Watson's algorithm
void TriangulateDelaunay(const Vec2* points, uint32_t numPoints, vector<Vec2>& outPoints, vector<uint32_t>& outTris)
{
	// calculate bounds
	Vec2r lower(FLT_MAX), upper(-FLT_MAX);

	for (uint32_t i=0; i < numPoints; ++i)
	{
		lower = Min(lower, Vec2r(points[i]));
		upper = Max(upper, Vec2r(points[i]));
	}

	Vec2r margin = Vec2r(upper-lower)*0.2f;

	Triangulation mesh(lower-margin, upper+margin);

	// insert all initial points into triangulation
	for (uint32_t i=0; i < numPoints; ++i)
		mesh.Insert(Vec2r(points[i]));

	// copy to output
	outPoints.resize(0);
	for (uint32_t i=3; i < mesh.vertices.size(); ++i)
		outPoints.push_back(Vec2(float(mesh.vertices[i].x), float(mesh.vertices[i].y)));
	
	outTris.resize(0);
	for (uint32_t i=0; i < mesh.triangles.size(); ++i)
	{
		const Triangle& t = mesh.triangles[i];

		// throw away tris connected to the initial bounding box 
		if (t.mVertices[0] < 3 || t.mVertices[1] < 3 || t.mVertices[2] < 3)
			continue;
	
		Vec2r a = mesh.vertices[t.mVertices[0]];
		Vec2r b = mesh.vertices[t.mVertices[1]];
		Vec2r c = mesh.vertices[t.mVertices[2]];

		if (TriArea(a, b, c) > 1.e-3f)
		{
			outTris.push_back(t.mVertices[0]-3);
			outTris.push_back(t.mVertices[1]-3);
			outTris.push_back(t.mVertices[2]-3);
		}
	}
}

#if 1
void RefineDelaunay(const Vec2* points, uint32_t numPoints, const uint32_t* triangles, uint32_t numTris, uint32_t maxPoints, float minAngle, float maxArea, std::vector<Vec2>& outPoints, std::vector<uint32_t>& outTris)
{
	// subdivide boundary segments
	Triangulation mesh(points, numPoints, triangles, numTris);		

	for (uint32_t i=0; i < 2; ++i)
	{
	// find boundary
	vector<Edge> edges;
	for (uint32_t i=0; i < mesh.triangles.size(); ++i)
	{
		for (uint32_t e=0; e < 3; ++e)
		{
			const Triangle& t = mesh.triangles[i];

			Edge edge(t.mVertices[e], t.mVertices[(e+1)%3]);
			vector<Edge>::iterator iter = find(edges.begin(), edges.end(), edge);
			
			if (iter != edges.end())
			{
				if (iter->mFaces[0] == uint32_t(-1))
				{
					iter->mFaces[0] = i;
				}
				else
				{
					assert(iter->mFaces[1] == uint32_t(-1));
					iter->mFaces[1] = i;
				}
			}
			else
			{
				edge.mFaces[0] = i;
				edges.push_back(edge);
			}
		}
	}	

	real shortestEdge = FLT_MAX;

	for (uint32_t i=0; i < edges.size(); ++i)
	{
		if (edges[i].mFaces[1] == uint32_t(-1))
		{
			real l = Length(mesh.vertices[edges[i].mIndices[0]]-mesh.vertices[edges[i].mIndices[1]]);

			if (l < shortestEdge)
				shortestEdge = l;
		}
	}
	 
	for (uint32_t i=0; i < edges.size(); ++i)
	{
		if (edges[i].mFaces[1] == uint32_t(-1))
		{
			Vec2r p = mesh.vertices[edges[i].mIndices[0]];
			Vec2r q = mesh.vertices[edges[i].mIndices[1]];	

			real l = Length(p-q);

			if (l > sqrtf(3.0f)*shortestEdge && mesh.vertices.size() < maxPoints)
			{
				mesh.Insert(real(0.5)*(p+q));	
				//assert(mesh.Valid() && "waa waa");
			}
		}
	}
	}
	// refine poor quality triangles
	while (mesh.vertices.size() < maxPoints)
	{
		Vec2r m;
		real r(0.0);
		if (0 && mesh.FindEncroachedSegment(m, r))
		{
			mesh.Insert(m);
		}
		else
		{
			// find poor quality triangle
			int t = mesh.FindPoorQualityTriangle(minAngle, maxArea);
		
			if (t != -1)
			{
				/*
				printf("refining %d\n", t);
				Vec2r a = mesh.vertices[mesh.triangles[t].mVertices[0]];
				Vec2r b = mesh.vertices[mesh.triangles[t].mVertices[1]];
				Vec2r c = mesh.vertices[mesh.triangles[t].mVertices[2]];

				printf("ab={{%f, %f}, {%f, %f}, {%f, %f}};\n", a.x, a.y, b.x, b.y, c.x, c.y);
				printf("r={%f, %f};\n", mesh.triangles[t].mCircumCenter.x, mesh.triangles[t].mCircumCenter.y); 
				*/


				Vec2r c = mesh.triangles[t].mCircumCenter;
#if DEBUG_PRINT
				printf("Circle[{%f, %f}, %f]\n", c.x, c.y, mesh.triangles[t].mCircumRadius);
#endif
				mesh.Insert(c);
			}
			else
			{
				break;
			}
		}
	}	

	outPoints.resize(0);
	for (uint32_t i=0; i < mesh.vertices.size(); ++i)
		outPoints.push_back(Vec2(float(mesh.vertices[i].x), float(mesh.vertices[i].y)));
	
	outTris.resize(0);
	for (uint32_t i=0; i < mesh.triangles.size(); ++i)
	{
	   outTris.push_back(mesh.triangles[i].mVertices[0]);	
	   outTris.push_back(mesh.triangles[i].mVertices[1]);	
	   outTris.push_back(mesh.triangles[i].mVertices[2]);	
	}
}
#endif


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

