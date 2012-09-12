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

	float2* mNewPositions;
	float* mNewDensities;
	Matrix22* mNewStress;

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

inline float norm(const Matrix22& m) { return sqrtf(Dot(m.cols[0], m.cols[0]) + Dot(m.cols[1], m.cols[1])); }
inline float sqr(float x) { return x*x; }
inline float kernel(float x) { return x; }//return x*sqr(max(1.0f-x*20.0f, 0.0f)); } 

inline void CollideCell(int cx, int cy, float2 xi, float ri,
	   	const unsigned int* cellStarts, const unsigned int* cellEnds, const unsigned int* indices,
		const float2* positions, const float* radii, float2& impulse, float& weight)
{
	const unsigned int cellIndex = GridHash(cx, cy);
	const unsigned int cellStart = cellStarts[cellIndex];
	const unsigned int cellEnd = cellEnds[cellIndex];
			
	// iterate over cell
	for (unsigned int i=cellStart; i < cellEnd; ++i)
	{
		unsigned int particleIndex = indices[i];
	
		const float2 xj = positions[particleIndex];
		const float rj = radii[particleIndex];

		// distance to sphere
		const float2 xij = xi - xj; 
		
		const float dSq = LengthSq(xij);
		const float rsum = ri + rj;
	
		if (dSq < sqr(rsum) && dSq > 0.001f)
		{
			const float d = sqrtf(dSq);
			const Vec2 n = xij / d;

			// project out of sphere
			impulse += 0.5f*kernel(rsum-d)*n;	

			weight += 1.0f;
		}
	}		
}



inline float2 Collide(
		float2 x,
		float r,
		const float2* positions,
		const float* radii,
		const float3* planes,
		int numPlanes,
		const unsigned int* cellStarts, 
		const unsigned int* cellEnds, 
		const unsigned int* indices)
{
	// collide particles
	int cx = GridCoord(x.x, invCellEdge);
	int cy = GridCoord(x.y, invCellEdge);
	
	float2 impulse;
	float weight = 0.0f;

	for (int i=cx-1; i <= cx+1; ++i)
	{
		for (int j=cy-1; j <= cy+1; ++j)
		{
			CollideCell(i, j, x, r, cellStarts, cellEnds, indices, positions, radii, impulse, weight);
		}
	}


	// collide planes
	for (int i=0; i < numPlanes; ++i)
	{
		float3 p = planes[i];
						
		// distance to plane
		float d = x.x*p.x + x.y*p.y - p.z;
			
		float mtd = d - r;
			
		if (mtd < 0.0f)
		{
			impulse -= mtd*float2(p.x, p.y);

			weight += 1.0f;
		}
	}

	if (weight > 0.0f)
		return impulse / weight;
	else
		return 0.0f;
}

void Integrate(int index, const float2* positions, float2* newPositions, float2* velocities, float2 gravity, float damp, float dt)
{
	// v += f*dt
	velocities[index] += (gravity - damp*velocities[index])*dt;

	// x += v*dt
	newPositions[index] = positions[index] + velocities[index]*dt;
}

void Update(GrainSystem s, float dt, float invdt)
{		
	for (int i=0; i < s.mNumGrains; ++i)
		Integrate(i, s.mPositions, s.mNewPositions, s.mVelocities, s.mParams.mGravity, s.mParams.mDamp, dt);

	memset(s.mCellStarts, 0, sizeof(unsigned int)*128*128);
	memset(s.mCellEnds, 0, sizeof(unsigned int)*128*128);
	
	ConstructGrid(invCellEdge, s.mNumGrains, s.mNewPositions, s.mIndices, s.mCellStarts, s.mCellEnds); 

	for (int k=0; k < 1; ++k)
	{
		for (int i=0; i < s.mNumGrains; ++i)
		{
			// solve position constraints
			
			float2 j = Collide(s.mNewPositions[i], s.mRadii[i],
				   	s.mNewPositions,
				   	s.mRadii,
				   	s.mParams.mPlanes,
				   	s.mParams.mNumPlanes, 
					s.mCellStarts, 
					s.mCellEnds,
				   	s.mIndices); 
	
			float2 x = s.mNewPositions[i] + j;

			s.mVelocities[i] = (x-s.mPositions[i])*invdt;

			s.mPositions[i] = x;
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

	s->mNewPositions = (float2*)malloc(numGrains*sizeof(float2));
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
	
	free(s->mNewPositions);
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
