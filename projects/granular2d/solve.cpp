#include <core/maths.h>

typedef Vec2 float2;
typedef Vec3 float3;

#include "solve.h"

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

struct GrainSystem
{
public:
	
	float2* mPositions;
	float2* mVelocities;
	float* mRadii;
	float* mDensities;
	Matrix22* mStress;

	float2* mNewVelocities;
	float* mNewDensities;
	Matrix22* mNewStress;

	unsigned int* mCellStarts;
	unsigned int* mCellEnds;
	unsigned int* mIndices;
	
	float mMaxRadius;
	
	int mNumGrains;
	GrainParams mParams;
};

bool doCollide = false;

float invCellEdge = 1.0f/0.1f;

// transform a world space coordinate into cell coordinate
unsigned int GridCoord(float x, float invCellEdge)
{
	// offset to handle negative numbers
	float l = x+1000.0f;
	
	int c = (unsigned int)(floorf(l*invCellEdge));
	
	return c;
}


unsigned int GridHash(int x, int y)
{
	const unsigned int kDim = 128;
	
	unsigned int cx = x & (kDim-1);
	unsigned int cy = y & (kDim-1);
	
	return cy*kDim + cx;
}

struct IndexPair
{
	IndexPair(unsigned int c, unsigned int p) : mCellId(c), mParticleId(p) {}
	
	unsigned int mCellId;
	unsigned int mParticleId;
	
	bool operator<(const IndexPair& a) const { return mCellId < a.mCellId; }
};

void ConstructGrid(float invCellEdge, int numGrains, const float2* positions, 
				   unsigned int* indices, unsigned int* cellStarts, unsigned int* cellEnds)
{	
	std::vector<IndexPair> indexPairs;
	indexPairs.reserve(numGrains);
	
	for (int i=0; i < numGrains; ++i)
	{
		indexPairs.push_back(IndexPair(GridHash(GridCoord(positions[i].x, invCellEdge),
												GridCoord(positions[i].y, invCellEdge)), i));
	}
	
	// sort the indices based on the cell id
	std::sort(indexPairs.begin(), indexPairs.end());
	
	// scan the particle-cell array to find the start and end
	for (int i=0; i < numGrains; ++i)
	{
		IndexPair c = indexPairs[i];
		
		if (i == 0)
		{
			cellStarts[c.mCellId] = i;
		}
		else
		{
			IndexPair p = indexPairs[i-1];

			if (c.mCellId != p.mCellId)
			{
				cellStarts[c.mCellId] = i;
				cellEnds[p.mCellId] = i;			
			}
		}
		
		if (i == numGrains-1)
		{
			cellEnds[c.mCellId] = i+1;
		}
		
		indices[i] = c.mParticleId;
	}
}

// calculate collision impulse
float2 CollisionImpulse(float2 va, float2 vb, float ma, float mb, float2 n, float d, float baumgarte, float overlap, float friction)
{
	// calculate relative velocity
	float2 vd = vb-va;
	
	// calculate relative normal velocity
	float vn = Dot(vd, n);
	
	/*
	const float kStiff = 20000.0f;
	const float kDamp = 100.0f;

	// total mass 
	float msum = ma + mb;

	if (vn < 0.0f)
	{
		return -(kStiff*d + kDamp*vn)*n*mb/msum;
	}
		
	return Vec2(0.0f, 0.0f);
	*/
	
	// calculate relative tangential velocity
	float2 vt = vd - n*vn;
	
	float rcpvt = 1.0f / sqrtf(Dot(vt, vt) + 0.0001f);

	// total mass 
	float msum = ma + mb;
		
	if (vn < 0.0f)
	{
		float bias = baumgarte*min(d+overlap, 0.0f);

		float2 jn = (vn + bias)*n;
		float2 jt = -max(friction*vn*rcpvt, -1.0f)*vt;
		
		return -(jn + jt)*mb/msum;

	}
	
	return Vec2(0.0f, 0.0f);
	
}

inline float norm(const Matrix22& m) { return sqrtf(Dot(m.cols[0], m.cols[0]) + Dot(m.cols[1], m.cols[1])); }
inline float sqr(float x) { return x*x; }
inline float kernel(float x) { return 1.0f; };//sqr(max(1.0f-x*10.0f, 0.0f)); } 

inline float2 CollideCell(unsigned int index, int cx, int cy, const unsigned int* cellStarts, const unsigned int* cellEnds, const unsigned int* indices,
				 const float2* positions, const float2* velocities, const float* radii, const float* densities, const Matrix22* stress, Matrix22& velocityGradient, float& newDensity, float baumgarte, float overlap)
{
	const float2 xi = positions[index];
	const float  ri = radii[index];

	const unsigned int cellIndex = GridHash(cx, cy);
	const unsigned int cellStart = cellStarts[cellIndex];
	const unsigned int cellEnd = cellEnds[cellIndex];
			
	// final impulse
	float2 j;

	// iterate over cell
	for (unsigned int i=cellStart; i < cellEnd; ++i)
	{
		unsigned int particleIndex = indices[i];
		
		if (particleIndex != index)
		{
			const float2 xj = positions[particleIndex];
			const float rj = radii[particleIndex];

			// distance to sphere
			const float2 xij = xi - xj; 
			
			const float dSq = LengthSq(xij);
			const float rsum = ri + rj;
		
			if (dSq < sqr(rsum))
			{
				const float d = sqrtf(dSq);
				const Vec2 n = xij / d;

				const Vec2 vi = velocities[index];
				const Vec2 vj = velocities[particleIndex];
			
				// inelastic collision impulse	
				Vec2 c = CollisionImpulse(vj, vi, 1.0f, 1.0f, n, d-rsum, baumgarte, overlap, 0.0f);

				//c += (vj-vi - 2.0f*c)*0.8f;

				if (!doCollide)
					c = Vec2();	

				// gradient of kernel function	
				float w = kernel(d);

				const Vec2 dw = n*w;

				// update the new velocity gradient 
				//if (c.x != 0.0f || c.y != 0.0f)
				if (densities[particleIndex] > 0.0f)
					velocityGradient += Outer(dw, vj-vi)*(1.0f/densities[particleIndex]);

				// apply forces due to last frames stress
				Vec2 f;	
			//	if (densities[index] > 0.0f && densities[particleIndex] > 0.0f)
			//		f = (stress[index]*-1.0f*sqr(1.0f/densities[index]) + stress[particleIndex]*sqr(1.0f/densities[particleIndex]))*dw;
			
				if (densities[particleIndex] > 0.0f)// && norm(stress[index]) > 0.2f)	
					f = stress[particleIndex]*(1.0f/densities[particleIndex])*dw;

				// apply collision and fiction impulses	
				j += c + f;

				newDensity += w;
			}
		}		
	}
	 
	return j;
}



inline void Collide(
		int index,
		const float2* positions,
		const float2* velocities,
		const float* radii,
		const float* densities,
		const Matrix22* stress,
		const float3* planes,
		int numPlanes,
		const unsigned int* cellStarts, 
		const unsigned int* cellEnds, 
		const unsigned int* indices, 
		float2* newVelocities, 
		float* newDensity, 
		Matrix22* newStress, 
		int numGrains, float baumgarte, float overlap)
{
	float2 x = positions[index];
	float2 v = velocities[index];
	float  r = radii[index];

	// collide particles
	int cx = GridCoord(x.x, invCellEdge);
	int cy = GridCoord(x.y, invCellEdge);
	
	Matrix22 velGrad; 
	float density = 0.0f;

	for (int i=cx-1; i <= cx+1; ++i)
	{
		for (int j=cy-1; j <= cy+1; ++j)
		{
			v += CollideCell(index, i, j, cellStarts, cellEnds, indices, positions, velocities, radii, densities, stress, velGrad, density, baumgarte, overlap);
		}
	}

	// calculate deviatoric stress from velocity gradient
	const Matrix22 strainRate = 0.5f*(velGrad + Transpose(velGrad));

	newStress[index] = strainRate*1.0f;
	//newStress[index] = newStress[index]-Trace(newStress[index])*Matrix22::Identity()*0.3f;
	
	newDensity[index] = density;

	// collide planes
	for (int i=0; i < numPlanes; ++i)
	{
		float3 p = planes[i];
						
		// distance to plane
		float d = x.x*p.x + x.y*p.y - p.z;
			
		float mtd = d - r;
			
		if (mtd < 0.0f)
		{
			v += CollisionImpulse(float2(0.0f, 0.0f), v, 0.0f, 1.0f, float2(p.x, p.y), mtd, baumgarte, overlap, 0.9f);
		}
	}

	// write back velocity
	newVelocities[index] = v;
}

void Integrate(int index, float2* positions, float2* velocities, float2 gravity, float damp, float dt)
{
	// v += f*dt
	velocities[index] += (gravity - damp*velocities[index])*dt;

	// x += v*dt
	positions[index] += velocities[index]*dt;
}

void Update(GrainSystem s, float dt, float invdt)
{		
	for (int i=0; i < s.mNumGrains; ++i)
		Integrate(i, s.mPositions, s.mVelocities, s.mParams.mGravity, s.mParams.mDamp, dt);

	memset(s.mCellStarts, 0, sizeof(unsigned int)*128*128);
	memset(s.mCellEnds, 0, sizeof(unsigned int)*128*128);
	
	ConstructGrid(invCellEdge, s.mNumGrains, s.mPositions, s.mIndices, s.mCellStarts, s.mCellEnds); 

	memcpy(s.mNewVelocities, s.mVelocities, sizeof(float)*2*s.mNumGrains);

	for (int k=0; k < 1; ++k)
	{
		doCollide = k==0;

		for (int i=0; i < s.mNumGrains; ++i)
		{
			Collide(i,
				   	s.mPositions,
				   	s.mVelocities,
				   	s.mRadii,
					s.mDensities,
					s.mStress,
				   	s.mParams.mPlanes,
				   	s.mParams.mNumPlanes, 
					s.mCellStarts, 
					s.mCellEnds,
				   	s.mIndices, 
					s.mNewVelocities, 
					s.mNewDensities,
					s.mNewStress, 
					s.mNumGrains,
				   	s.mParams.mBaumgarte*invdt, 
					s.mParams.mOverlap);
		}

		for (int i=0; i < s.mNumGrains; ++i)
		{
			s.mVelocities[i] = s.mNewVelocities[i];
			s.mStress[i] = s.mNewStress[i];
			s.mDensities[i] = s.mNewDensities[i];
		}
	}
}


//------------------------------------------------------------------


GrainSystem* grainCreateSystem(int numGrains)
{
	GrainSystem* s = new GrainSystem();
	
	s->mNumGrains = numGrains;
	
	s->mPositions = (float2*)malloc(numGrains*sizeof(float2));
	s->mVelocities = (float2*)malloc(numGrains*sizeof(float2));
	s->mRadii = (float*)malloc(numGrains*sizeof(float));
	s->mDensities = (float*)malloc(numGrains*sizeof(float));
	s->mStress = (Matrix22*)malloc(numGrains*sizeof(Matrix22));

	s->mNewVelocities = (float2*)malloc(numGrains*sizeof(float2));
	s->mNewDensities = (float*)malloc(numGrains*sizeof(float));
	s->mNewStress = (Matrix22*)malloc(numGrains*sizeof(Matrix22));

	memset(s->mDensities, 0, numGrains*sizeof(float));	
	memset(s->mStress, 0, numGrains*sizeof(Matrix22));
	
	s->mCellStarts = (unsigned int*)malloc(128*128*sizeof(unsigned int));
	s->mCellEnds = (unsigned int*)malloc(128*128*sizeof(unsigned int));
	s->mIndices = (unsigned int*)malloc(numGrains*sizeof(unsigned int));

	return s;
}

void grainDestroySystem(GrainSystem* s)
{
	free(s->mPositions);
	free(s->mVelocities);
	free(s->mRadii);
	free(s->mStress);
	free(s->mDensities);
	
	free(s->mNewVelocities);
	free(s->mNewStress);
	free(s->mNewDensities);

	free(s->mCellStarts);
	free(s->mCellEnds);
	free(s->mIndices);

	delete s;
}

void grainSetPositions(GrainSystem* s, float* p, int n)
{
	memcpy(&s->mPositions[0], p, sizeof(float2)*n);
}

void grainSetVelocities(GrainSystem* s, float* v, int n)
{
	memcpy(&s->mVelocities[0], v, sizeof(float2)*n);	
}

void grainSetRadii(GrainSystem* s, float* r)
{
	memcpy(&s->mRadii[0], r, sizeof(float)*s->mNumGrains);
}

void grainGetPositions(GrainSystem* s, float* p)
{
	memcpy(p, &s->mPositions[0], sizeof(float2)*s->mNumGrains);
}

void grainGetVelocities(GrainSystem* s, float* v)
{
	memcpy(v, &s->mVelocities[0], sizeof(float2)*s->mNumGrains);
}

void grainGetRadii(GrainSystem* s, float* r)
{
	memcpy(r, &s->mRadii[0], sizeof(float)*s->mNumGrains);
}

void grainSetParams(GrainSystem* s, GrainParams* params)
{
	//cudaMemcpy(s->mParams, params, sizeof(GrainParams), cudaMemcpyHostToDevice);
	s->mParams = *params;
}

void grainUpdateSystem(GrainSystem* s, float dt, int iterations, GrainTimers* timers)
{
	dt /= iterations;
			
	const float invdt = 1.0f / dt;
	
	for (int i=0; i < iterations; ++i)
	{
		Update(*s, dt, invdt);
	}	
}
