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
	float2* mnewVelocities;
	float* mRadii;
	float2* mStressRow1;
	float2* mStressRow2;
	
	unsigned int* mCellStarts;
	unsigned int* mCellEnds;
	unsigned int* mIndices;
	
	float mMaxRadius;
	
	int mNumGrains;
	GrainParams mParams;
};

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

void IntegrateForce(int index, float2* velocities, float2* newVelocities, float2 gravity, float damp, float dt)
{
	// v += f*dt
	velocities[index] += (gravity - damp*velocities[index])*dt;
}

float inline sqr(float x) { return x*x; }

float2 CollideCell(int index, int cx, int cy, const unsigned int* cellStarts, const unsigned int* cellEnds, const unsigned int* indices,
				 const float2* positions, const float2* velocities, const float* radii, const float2* stress1, const float2* stress2, float2* newStress, float baumgarte, float overlap)
{
	float2 x = positions[index];
	float  r = radii[index];
	float2 j;
	
	unsigned int cellIndex = GridHash(cx, cy);
	
	unsigned int cellStart = cellStarts[cellIndex];
	unsigned int cellEnd = cellEnds[cellIndex];
			
	//newStress[0] = Vec2();
	//newStress[1] = Vec2();
	
	for (unsigned int i=cellStart; i < cellEnd; ++i)
	{
		unsigned int particleIndex = indices[i];
		
		if (particleIndex != index)
		{
			// distance to sphere
			float2 t = x - positions[particleIndex];
			
			float d = LengthSq(t);
			float rsum = r + radii[particleIndex];
			float mtd = d - sqr(rsum);
			
			if (mtd < 0.0f)
			{
				Vec2 n;
						
				d = sqrtf(d);
				n = t / d;
				
				Vec2 s = CollisionImpulse(velocities[particleIndex], velocities[index] , 1.0f, 1.0f, n, d-rsum , baumgarte, overlap, 0.0f);
				//j += s;
				
				// update this particles stress tensor
				if (s.x != 0.0f || s.y != 0.0f)
				{
					newStress[0] += t.x*velocities[particleIndex];
					newStress[1] += t.y*velocities[particleIndex];
				}
												
				Vec2 f = Vec2(Dot(n, stress1[particleIndex]), Dot(n, stress2[particleIndex]))*0.8f;

				j += s + f;
			}
		}		
	}
	
	/*
	float d = 0.5f*(newStress[0].y + newStress[1].x);
	newStress[0].y = d;
	newStress[1].x = d;
	
	
	for (unsigned int i=cellStart; i < cellEnd; ++i)
	{
		unsigned int particleIndex = indices[i];
		
		if (particleIndex != index)
		{
			// distance to sphere
			float2 t = x - positions[particleIndex];
			
			float d = LengthSq(t);
			float rsum = r + radii[particleIndex];
			float mtd = d - sqr(rsum);
			
			if (mtd < 0.0f)
			{
				
				j -= Vec2(Dot(t, newStress[0]), Dot(t, newStress[1]))*0.5f/sqrtf(d);
			}
		}
	}
		 */
	 
	return j;
}



void Collide(int index, const float2* positions, const float2* velocities, const float* radii, const float3* planes, int numPlanes,
			 const unsigned int* cellStarts, const unsigned int* cellEnds, const unsigned int* indices, float2* newVelocities, float2* stress1, float2* stress2, int numGrains, float baumgarte, float overlap)
{
	float2 x = positions[index];
	float2 v = velocities[index];
	float  r = radii[index];

	// collide particles
	int cx = GridCoord(x.x, invCellEdge);
	int cy = GridCoord(x.y, invCellEdge);
	
	Vec2 newStress[2];

	for (int i=cx-1; i <= cx+1; ++i)
	{
		for (int j=cy-1; j <= cy+1; ++j)
		{
			v += CollideCell(index, i, j, cellStarts, cellEnds, indices, positions, velocities, radii, stress1, stress2, newStress, baumgarte, overlap);
		}
	}
	
	float d = 0.5f*(newStress[0].y + newStress[1].x);
	newStress[0].y = d;
	newStress[1].x = d;

	/*
	for (int i=cx-1; i <= cx+1; ++i)
	{
		for (int j=cy-1; j <= cy+1; ++j)
		{
			v += CollideCellF(index, i, j, cellStarts, cellEnds, indices, positions, velocities, radii, stress1, stress2, newStress, baumgarte, overlap);
		}
	}
	 */	
	// write back new stress tensor
	stress1[index] = newStress[0];
	stress2[index] = newStress[1];

	
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

void IntegrateVelocity(int index, float2* positions, float2* velocities, float2* newVelocities, float dt)
{
	// x += v*dt
	velocities[index] = newVelocities[index];
	positions[index] += velocities[index]*dt;
}

void Update(GrainSystem s, float dt, float invdt)
{		
	for (int i=0; i < s.mNumGrains; ++i)
	{
		IntegrateForce(i, s.mVelocities, s.mnewVelocities, s.mParams.mGravity, s.mParams.mDamp, dt);
	}

	memset(s.mCellStarts, 0, sizeof(unsigned int)*128*128);
	memset(s.mCellEnds, 0, sizeof(unsigned int)*128*128);
	
	ConstructGrid(invCellEdge, s.mNumGrains, s.mPositions, s.mIndices, s.mCellStarts, s.mCellEnds); 
	
	for (int i=0; i < s.mNumGrains; ++i)
	{
		Collide(i, s.mPositions, s.mVelocities, s.mRadii, s.mParams.mPlanes, s.mParams.mNumPlanes, 
				s.mCellStarts, s.mCellEnds, s.mIndices, s.mnewVelocities, s.mStressRow1, s.mStressRow2, s.mNumGrains, s.mParams.mBaumgarte*invdt, s.mParams.mOverlap);
	}

	for (int i=0; i < s.mNumGrains; ++i)
	{
		IntegrateVelocity(i, s.mPositions, s.mVelocities, s.mnewVelocities, dt); 		
	}
}


//------------------------------------------------------------------


GrainSystem* grainCreateSystem(int numGrains)
{
	GrainSystem* s = new GrainSystem();
	
	s->mNumGrains = numGrains;
	
	s->mPositions = (float2*)malloc(numGrains*sizeof(float2));
	s->mVelocities = (float2*)malloc(numGrains*sizeof(float2));
	s->mnewVelocities = (float2*)malloc(numGrains*sizeof(float2));
	s->mRadii = (float*)malloc(numGrains*sizeof(float));
	s->mStressRow1 = (float2*)malloc(numGrains*sizeof(float2));
	s->mStressRow2 = (float2*)malloc(numGrains*sizeof(float2));
	
	memset(s->mStressRow1, 0, numGrains*sizeof(float2));
	memset(s->mStressRow2, 0, numGrains*sizeof(float2));
	
	
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
	free(s->mnewVelocities);
	free(s->mStressRow1);
	free(s->mStressRow2);
	
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
		memset(s->mnewVelocities, 0, sizeof(float)*2*s->mNumGrains);
		Update(*s, dt, invdt);
	}	
}