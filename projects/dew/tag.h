#pragma once

#include "core/maths.h"
#include <vector>

struct Vertex
{
	Vertex() {}
	Vertex(Point3 p, Vec3 n) : position(p), normal(n) {}

	Point3 position;
	Vec3 normal;
};

void SquareBrush(float t, std::vector<Vertex>& verts, float w, float h)
{
	const Vertex shape[] = 
	{
		Vertex(Point3(-w,  h, 0.0f), Vec3(-1.0f, 0.0f, 0.0f)),
		Vertex(Point3(-w,  -h, 0.0f), Vec3(-1.0f, 0.0f, 0.0f)),
	
		Vertex(Point3( -w, -h, 0.0f), Vec3( 0.0f, -1.0f, 0.0f)),
		Vertex(Point3( w,  -h, 0.0f), Vec3( 0.0f, -1.0f, 0.0f)),
	
		Vertex(Point3( w, -h, 0.0f), Vec3( 1.0f, 0.0f, 0.0f)),
		Vertex(Point3( w,  h, 0.0f), Vec3( 1.0f, 0.0f, 0.0f)),
		
		Vertex(Point3( w,  h, 0.0f), Vec3( 0.0f, 1.0f, 0.0f)),
		Vertex(Point3( -w, h, 0.0f), Vec3( 0.0f, 1.0f, 0.0f))
	};

	verts.assign(shape, shape+8);

}
struct Tag
{
	Tag(float smoothing, float width, float height) : basis(Matrix44::kIdentity), smoothing(smoothing), width(width), height(height), draw(false) 
	{
		samples.reserve(4096);
		vertices.reserve(100000);
		indices.reserve(100000);
	}

	void Start()
	{
		OutputCap(true);	

		draw = true;
	}

	void Stop()
	{
		OutputCap(false);

		draw = false;
	}

	void OutputCap(bool flip)
	{
		// draw cap
		SquareBrush(1.f, brush, width, height);

		Point3 center(0.0f);

		// transform verts and create faces
		for (size_t i=0; i < brush.size(); ++i)		
		{
			center += Vec3(brush[i].position);
		}
		
		center /= brush.size();
	
		uint32_t i0 = vertices.size();
		
		float dir = flip?-1.0f:1.0f;
		Vec3 n = dir*Vec3(basis.GetCol(2));

		Vertex c(basis*center, n);
		vertices.push_back(c);

		// transform verts and create faces
		for (size_t i=0; i < brush.size(); ++i)
		{
			// transform position and normal to world space
			Vertex v(basis*brush[i].position, n);

			vertices.push_back(v);
			
			if (i > 0)
			{
				if (!flip)
				{
					indices.push_back(i0);
					indices.push_back(i0+i);
					indices.push_back(i0+i+1);
				}
				else
				{
					indices.push_back(i0+i+1);
					indices.push_back(i0+i);
					indices.push_back(i0);
				}
			}	
		}
	}

	void PushSample(float t, Matrix44 m)
	{
		// evaluate brush
		SquareBrush(t, brush, width, height);

		size_t startIndex = vertices.size();

		Point3 prevPos = samples.empty()?m.GetTranslation():samples.back();
		Point3 curPos = m.GetTranslation();

		// low-pass filter position
		Point3 p = Lerp(prevPos, curPos, 1.0f-smoothing);
		m.SetTranslation(p);

		samples.push_back(m.GetTranslation());

		// need at least 4 points to construct valid tangents
		if (samples.size() < 4)
			return;

		// the point we are going to output
		size_t c = samples.size()-3;

		// calculate the tangents for the two samples using central differencing
		Vec3 tc = Normalize(samples[c+1]-samples[c-1]);
		Vec3 td = Normalize(samples[c+2]-samples[c]);
		float a = acosf(Dot(tc, td));

		if (fabsf(a) > 0.001f)
		{
			// use the parallel transport method to move the reference frame along the curve
			Vec3 n = Normalize(Cross(tc, td));

			if (samples.size() == 4)
				basis = TransformFromVector(Normalize(tc));
		
			// 'transport' the basis forward
			basis = RotationMatrix(a, n)*basis;
			
			m = basis;
			m.SetTranslation(samples[c]);
			basis = m;
		}
	
		if (!draw)
			return;
		
		// transform verts and create faces
		for (size_t i=0; i < brush.size(); ++i)
		{
			// transform position and normal to world space
			Vertex v(m*brush[i].position, m*brush[i].normal);

			vertices.push_back(v);
		}

		if (startIndex != 0)
		{
			size_t b = brush.size();

			for (size_t i=0; i < b; ++i)
			{
				size_t curIndex = startIndex + i;
				size_t nextIndex = startIndex + (i+1)%b; 

				indices.push_back(curIndex);
				indices.push_back(curIndex-b);
				indices.push_back(nextIndex-b);
				
				indices.push_back(nextIndex-b);
				indices.push_back(nextIndex);
				indices.push_back(curIndex);			
			}	
		}
	}

	void Draw()
	{
		if (vertices.empty())
			return;

		// draw the tag
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, sizeof(Vertex), &vertices[0].position);
		glEnableClientState(GL_NORMAL_ARRAY);
		glNormalPointer(GL_FLOAT, sizeof(Vertex), &vertices[0].normal);

		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, &indices[0]);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);

	}

	void ExportToObj(const char* path)
	{
		FILE* f = fopen(path, "w");

		if (!f)
			return;

		fprintf(f, "# %d positions\n", int(vertices.size()));

		for (uint32_t i=0; i < vertices.size(); ++i)
		{
			Point3 p = vertices[i].position;
			fprintf(f, "v %f %f %f\n", p.x, p.y, p.z);
		}

		fprintf(f, "# %d normals\n", int(vertices.size()));

		for (uint32_t i=0; i < vertices.size(); ++i)
		{
			Vec3 n = vertices[i].normal;
			fprintf(f, "vn %f %f %f\n", n.x, n.y, n.z);
		}

		fprintf(f, "# %d faces\n", int(indices.size()/3));

		for (uint32_t t=0; t < indices.size(); t+=3)
		{
			// obj is 1 based
			uint32_t i = indices[t+0]+1;
			uint32_t j = indices[t+1]+1;
			uint32_t k = indices[t+2]+1;

			fprintf(f, "f %d//%d %d//%d %d//%d\n", i, i, j, j, k, k); 
		}	

		fclose(f);
	}

	void GetBounds(Point3& lower, Point3& upper)
	{
		Point3 l(FLT_MAX);
		Point3 u(-FLT_MAX);

		for (uint32_t i=0; i < samples.size(); ++i)
		{
			l = Min(l, samples[i]);
			u = Max(u, samples[i]);	
		}	

		lower = l;
		upper = u;
	}

	void Clear()
	{
		samples.resize(0);
		brush.resize(0);
		vertices.resize(0);
		indices.resize(0);
	}

	Matrix44 basis;

	std::vector<Point3> samples;
	std::vector<Vertex> brush;
	std::vector<Vertex> vertices;
	std::vector<uint32_t> indices;

	float smoothing;
	float width;
	float height;
	bool draw;
};


