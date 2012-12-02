#if 1

#include <core/maths.h>
#include <core/shader.h>

typedef Vec2 float2;
typedef Vec3 float3;

#include "solve.h"

#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

// Todo List 
// -------------------
//
// * Unify artificial pressure and constraint function 
// * Density analysis
// 		- look at using a higher order solver for the density constraint 
// 		- and get rid of the precomputed scaling factor (done, use regularization)
// * CPU Jacobi solver (done)
// * GPU Jacobi solver 
// * Port to 3d 
// * No-slip boundary (done)
// * Signed distance field collision
// * Vorticity confinement (done)
// * Stable viscosity
// * Improved boundary density estimate

const int kMaxContactsPerSphere = 40; 

struct GrainSystem
{
public:
	
	float2* mPositions;
	float2* mVelocities;
	float* mRadii;
	float* mDensity;
	float* mNearDensity;
	float* mVorticity;

	float2* mCandidatePositions;
	float2* mCandidateVelocities;

	float2* mForces;

	Matrix22* mStress;

	unsigned int* mCellStarts;
	unsigned int* mCellEnds;
	unsigned int* mIndices;

	uint32_t* mSpringIndices;
	float* mSpringLengths;
	int mNumSprings;

	int* mContacts;
	int* mContactCounts;	

	float mMaxRadius;
	
	int mNumGrains;
	GrainParams mParams;
};

float kRadius = 0.2f;
float kInvCellEdge = 1.0f/0.4f;

// transform a world space coordinate into cell coordinate
unsigned int GridCoord(float x)
{
	// offset to handle negative numbers
	float l = x+1000.0f;
	
	int c = (unsigned int)(floorf(l*kInvCellEdge));
	
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
		indexPairs.push_back(IndexPair(GridHash(GridCoord(positions[i].x),
												GridCoord(positions[i].y)), i));
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

inline float sqr(float x) { return x*x; }
inline float cube(float x) { return x*x*x; }

/*
inline float W(float r, float h)
{
	const float k = 15.0f/(14.0f*kPi); // normalization factor from Monaghan 2005

	float q = r/h;
	float m = 0.0f; 
	if (q < 1.0f)
		m = (cube(2.0f-q) - 4.0f*cube(1.0f-q))*k;
	else if (q < 2.0f)
		m = cube(2.0f-q)*k;

	return m/sqr(h);
}

inline float dWdx(float r, float h)
{
	const float k = 15.0f/(14.0f*kPi); // normalization factor from Monaghan 2005

	float q = r/h;
	float m = 0.0f; 
	if (q < 1.0f)
		m = -k*(3.0f*sqr(2.0f-q) - 12.0f*sqr(1.0f-q))/h;
	else if (q < 2.0f)
		m = -k*3.0f*sqr(2.0f-q)/h;

	return m/sqr(h);
}

*/

/*
 * Poly6
 *
 */

/*
inline float W(float r, float h)
{
	float k = 4.0f/(kPi*h*h*h*h*h*h*h*h);

	if (r < h)
		return k*cube(h*h - r*r);
	else
		return 0.0f;
}

inline float dWdx(float r, float h)
{
	float k = 4.0f/(kPi*h*h*h*h*h*h*h*h);

	if (r < h)
		return -k*6.0f*sqr(h*h - r*r);
	else
		return 0.0f;
}
*/
/*
 * Spiky kernel
 *
 */
inline float W(float r, float h)
{
	float k = 6.0f/(kPi*h*h);

	if (r < h)
		return k*sqr(1.0f-r/h);
	else
		return 0.0f;
}
inline float dWdx(float r, float h)
{
	float k = -12.0f/(kPi*h*h*h);

	if (r < h)
		return k*(1.0f-r/h);
	else
		return 0.0f;
}	

inline float Wnear(float r, float h)
{
	float k = 6.0f/(kPi*h*h);

	if (r < h)
		return k*cube(1.0f-r/h);
	else
		return 0.0f;
}


inline float dWNeardx(float r, float h)
{
	float k = -18.0f/(kPi*h*h*h);

	if (r < h)
		return k*sqr(1.0f-r/h);
	else
		return 0.0f;
}

inline int Collide(
		int index, 
		const unsigned int* cellStarts, 
		const unsigned int* cellEnds, 
		const unsigned int* indices,
		const float2* positions,
		float h,
		int* contacts,
		int maxContacts) 
{
	const float2 xi = positions[index];

	// collide particles
	int cx = GridCoord(xi.x);
	int cy = GridCoord(xi.y);
	
	int numContacts = 0;
	
	for (int i=cx-1; i <= cx+1; ++i)
	{
		for (int j=cy-1; j <= cy+1; ++j)
		{
			const unsigned int cellIndex = GridHash(i, j);
			const unsigned int cellStart = cellStarts[cellIndex];
			const unsigned int cellEnd = cellEnds[cellIndex];
					
			// iterate over cell
			for (unsigned int i=cellStart; i < cellEnd; ++i)
			{
				const unsigned int particleIndex = indices[i];

				if (int(particleIndex) == index)
					continue;
				
				const float2 xj = positions[particleIndex];

				// distance to sphere
				const float2 xij = xi - xj; 
				
				const float dSq = LengthSq(xij);
			
				if (dSq < sqr(h))
				{	
					contacts[numContacts++] = particleIndex;	

					if (numContacts == maxContacts)
						return numContacts;
				}	
			}
		}
	}

	return numContacts;
}

inline float CalculateDensity(
		int index,
		const float2* positions,
		const float2* velocities,		
		const int* contacts, 
		int numContacts,
		const float3* planes,
		int numPlanes,
		float h,
		float mass,
		float restDensity,
		float& nearRho)
{
	float2 xi = positions[index];
	
	float rho = 0.0f;

	float s = 0.0f; 
	float2 sv;

	for (int i=0; i < numContacts; ++i)
	{
		const int particleIndex = contacts[i];

		const float2 xj = positions[particleIndex];
		const float2 xij = xi-xj;
		
		const float dSq = LengthSq(xij);
	
		if (dSq < sqr(h))
		{
			const float d = sqrtf(dSq);
			const float w = W(d, h);

			const float2 dw = 1.0f/restDensity*dWdx(d,h)*xij/d; 
			s += Dot(dw, dw);
			sv += dw;

			rho += max(w, 0.0f);
		}
	}

	// constraint regularization
	nearRho = s + Dot(sv, sv) + 300.0f; 

/*	
	// collide planes
	for (int i=0; i < numPlanes; ++i)
	{
		float3 p = planes[i];

		// distance to plane
		float d = xi.x*p.x + xi.y*p.y - p.z;
		float mtd = d-h;
			
		if (mtd <= 0.0f)
		{
			const float w = W(max(d, 0.0f), h);
			rho += mass*max(w, 0.0f);
		}
	}
*/	
	//printf("%f\n", rho);

	return rho;
}

inline void SolvePositions(
		int index,
		float2* positions,
		const float2* velocities,		
		const float* densities,
		const float* nearDensities,
		float2* forces,
		const int* contacts, 
		int numContacts,
		const float3* planes,
		int numPlanes,
		float h,
		float mass,
		float dt,
		float restDensity)
{
	float2 xi = positions[index];
	
	// collide particles
	float2 pressureForce;
	float2 delta;

	float2 cw = 0.0f;

	float rho = densities[index];
   	float rhoNear = nearDensities[index];

	// scaling factor based on a filled neighbourhood
	const float s = 1.0f/rhoNear;//1.0f/600.f;

	//printf("%f\n", rhoNear);

	float m = 0.0f;
	float2 mv;

	// apply position updates
	for (int i=0; i < numContacts; ++i)
	{
		const int particleIndex = contacts[i];

		const float2 xj = positions[particleIndex];
		const float2 xij = xi-xj;
		
		const float dSq = LengthSq(xij);

		if (dSq < sqr(h) && dSq > 0.0f)
		{
			float d = sqrtf(dSq);

			float2 dw = 1.0f/restDensity*dWdx(d,h)*xij/d; 
			float j = s*(rho + densities[particleIndex] + 0.04f*(sqr(W(d,h)/W(h*0.3f, h))));

			// don't allow position to move more than the smoothing kernel length
			//if (fabsf(j) > h*0.005f)
			//	j = h*Sign(j)*0.005f;

			m += Dot(dw, dw);
			mv += dw;

			forces[index] -= j*dw;
		}
	}
	
	static float mmax = 0.0f;
	float mc = m + Dot(mv, mv);
	if (mc > mmax)
	{
		mmax = mc;
		//printf("%f\n", mc);
	}

	// collide planes
	for (int i=0; i < numPlanes; ++i)
	{
		float3 p = planes[i];

		xi = positions[index];

		// distance to plane
		float d = xi.x*p.x + xi.y*p.y - p.z;
		float mtd = d-h*0.5f;
			
		if (mtd <= 0.0f)
		{
			const float2 n = float2(p.x, p.y);
			delta -= mtd*n;

			//positions[index] -= mtd*n;
			forces[index] -= mtd*n;
		}
	}
}

inline float2 SolveVelocities(
		int index,
		float2* positions,
		const float2* velocities,		
		const float* densities,
		float* vorticities,
		float2* forces,
		const int* contacts, 
		int numContacts,
		const float3* planes,
		int numPlanes,
		float h,
		float mass,
		float dt,
		float restDensity)
{
	float2 xi = positions[index];
	float2 delta;
	float vorticity = 0.0f;
	float2 vorticityGrad;

	//float kSurfaceTension = 0.05f;
	float kViscosity = 0.02f*dt;

	for (int i=0; i < numContacts; ++i)
	{
		const int particleIndex = contacts[i];

		const float2 xj = positions[particleIndex];
		const float2 xij = xi-xj;
	
		const float dSq = LengthSq(xij);

		if (dSq < sqr(h) && dSq > 0.0f)
		{
			float d = sqrtf(dSq);
			float w = W(d, h);
			float2 dw = dWdx(d, h)*xij/d;
			
			assert(isfinite(d));
			assert(isfinite(dw.x));
			assert(isfinite(dw.y));

			const float2 vij = (velocities[particleIndex]-velocities[index]);

			vorticity += Cross(dw, vij); 
			vorticityGrad += fabsf(vorticities[particleIndex])*dw;

			//delta -= kSurfaceTension*xij*w;
			delta += kViscosity*vij*w; 
		}
	}

	// vorticity confinement - see Bubbles Alive for prior art :(
//	delta -= dt*100.5f*PerpCCW(SafeNormalize(vorticityGrad))*Sign(vorticity)/restDensity;	

	vorticities[index] = vorticity;

	/*
	// collide planes
	for (int i=0; i < numPlanes; ++i)
	{
		float3 p = planes[i];

		xi = positions[index];

		// distance to plane
		float d = xi.x*p.x + xi.y*p.y - p.z;
		float mtd = d-h;
			
		if (mtd <= 0.0f)
		{
			const float2 n = float2(p.x, p.y);
			float j = Dot(velocities[index], n);

			if (j > 0.0f)
				delta -= velocities[index]*0.2f;
		}
	}
	*/

	return delta;
}

void Integrate(int index, const float2* positions, float2* candidatePositions, float2* velocities, float2 gravity, float damp, float dt)
{
	// v += f*dt
	velocities[index] += (gravity - damp*velocities[index])*dt;

	// x += v*dt
	candidatePositions[index] = positions[index] + velocities[index]*dt;
}

void Update(GrainSystem s, float dt, float invdt)
{		
	for (int i=0; i < s.mNumGrains; ++i)
		Integrate(i, s.mPositions, s.mCandidatePositions, s.mVelocities, s.mParams.mGravity, s.mParams.mDamp, dt);
	
	float kMass = s.mParams.mMass;
	float kRestDensity = s.mParams.mRestDensity;

	memset(s.mCellStarts, 0, sizeof(unsigned int)*128*128);
	memset(s.mCellEnds, 0, sizeof(unsigned int)*128*128);

	ConstructGrid(kInvCellEdge, s.mNumGrains, s.mCandidatePositions, s.mIndices, s.mCellStarts, s.mCellEnds); 

	// find neighbours
	for (int i=0; i < s.mNumGrains; ++i)
	{
		s.mContactCounts[i] = Collide(i,
									  s.mCellStarts,
									  s.mCellEnds,
									  s.mIndices,
									  s.mCandidatePositions,
									  kRadius,
									  &s.mContacts[i*kMaxContactsPerSphere],
									  kMaxContactsPerSphere);

	}

	const int kNumPositionIterations = 5;

	for (int i=0; i < s.mNumGrains; ++i)
	{
		s.mDensity[i] = 0.0f;
		s.mNearDensity[i] = 0.0f;
		s.mForces[i] = 0.0f;
	}

	float maxDensity;
	float avgDensity;
	int k = 0;

	while (k++ < kNumPositionIterations)// || maxDensity > kRestDensity)
	{
		maxDensity = 0.0f;
		avgDensity = 0.0f;

		// calculate predicted density
		for (int i=0; i < s.mNumGrains; ++i)
		{
			float rho = 0.0f;
			float nearRho = 0.0f;

			rho = CalculateDensity(
					i,
				   	s.mCandidatePositions,
					s.mVelocities,				   	
					&s.mContacts[i*kMaxContactsPerSphere],
					s.mContactCounts[i],
				   	s.mParams.mPlanes,
				   	s.mParams.mNumPlanes,					
					kRadius,
					kMass,
					kRestDensity,
					nearRho);

			s.mForces[i] = 0.0f;

			s.mDensity[i] = rho/kRestDensity-1.0f;// + 0.2f*sqr(rho/W(kRadius*0.3f, kRadius));
			s.mNearDensity[i] = nearRho;//kNearStiffness*nearRho/kRestDensity;

			maxDensity = max(maxDensity, rho);
			avgDensity += rho;
		}

		avgDensity /= s.mNumGrains;

		//printf("%f %f\n", maxDensity, avgDensity);

		// calculate pressure forces
		for (int i=0; i < s.mNumGrains; ++i)
		{
			SolvePositions(
					i,
				   	s.mCandidatePositions,
					s.mVelocities,				   	
					s.mDensity,
					s.mNearDensity,
					s.mForces,
					&s.mContacts[i*kMaxContactsPerSphere],
					s.mContactCounts[i],
				   	s.mParams.mPlanes,
				   	s.mParams.mNumPlanes,					
					kRadius,
					kMass,
					dt,
					kRestDensity);
		}

		for (int i=0; i < s.mNumGrains; ++i)
		{
			s.mCandidatePositions[i] += s.mForces[i];///s.mContactCounts[i];	
		}
		/*
		for (int i=0; i < s.mNumGrains; ++i)
		{
			s.mCandidatePositions[i] = s.mCandidatePositions[i]+s.mForces[i];///max(float(s.mContactCounts[i]), 1.0f);
		}
		*/
	
	}
	
	for (int i=0; i < s.mNumGrains; ++i)
	{
		s.mVelocities[i] = (s.mCandidatePositions[i]-s.mPositions[i])*invdt;


		s.mForces[i] = SolveVelocities(
					i,
				   	s.mCandidatePositions,
					s.mVelocities,				   	
					s.mDensity,
					s.mVorticity,
					s.mForces,
					&s.mContacts[i*kMaxContactsPerSphere],
					s.mContactCounts[i],
				   	s.mParams.mPlanes,
				   	s.mParams.mNumPlanes,					
					kRadius,
					kMass,
					dt,
					kRestDensity);
					
					
	}	

	for (int i=0; i < s.mNumGrains; ++i)
	{
		s.mVelocities[i] += s.mForces[i];
		s.mPositions[i] = s.mCandidatePositions[i];//s.mVelocities[i]*dt;
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
	s->mDensity = (float*)malloc(numGrains*sizeof(float));
	s->mNearDensity = (float*)malloc(numGrains*sizeof(float));
	s->mVorticity = (float*)malloc(numGrains*sizeof(float));

	s->mForces = (float2*)malloc(numGrains*sizeof(float2));

	s->mContacts = (int*)malloc(numGrains*kMaxContactsPerSphere*sizeof(int));
	s->mContactCounts = (int*)malloc(numGrains*sizeof(int));

	s->mCandidatePositions = (float2*)malloc(numGrains*sizeof(float2));
	s->mCandidateVelocities = (float2*)malloc(numGrains*sizeof(float2));

	for (int i=0; i < s->mNumGrains; ++i)
	{
		s->mVelocities[i] = 0.0f;
		s->mPositions[i] = 0.0f;

		s->mDensity[i] = 0.0f;
		s->mNearDensity[i] = 0.0f;
		s->mVorticity[i] = 0.0f;
	}

	s->mCellStarts = (unsigned int*)malloc(128*128*sizeof(unsigned int));
	s->mCellEnds = (unsigned int*)malloc(128*128*sizeof(unsigned int));
	s->mIndices = (unsigned int*)malloc(numGrains*sizeof(unsigned int));

	s->mSpringIndices = NULL;
	s->mSpringLengths = NULL;
	s->mNumSprings = 0;

	return s;
}

void grainDestroySystem(GrainSystem* s)
{
	free(s->mPositions);
	free(s->mVelocities);
	free(s->mRadii);
	free(s->mDensity);
	free(s->mNearDensity);
	free(s->mVorticity);
	
	free(s->mForces);

	free(s->mContacts);
	free(s->mContactCounts);
	
	free(s->mCandidateVelocities);
	free(s->mCandidatePositions);

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

void grainSetSprings(GrainSystem* s, const uint32_t* springIndices, const float* springLengths, uint32_t numSprings)
{
	s->mSpringIndices = (uint32_t*)malloc(numSprings*2*sizeof(uint32_t));
	s->mSpringLengths = (float*)malloc(numSprings*sizeof(float));

	memcpy(s->mSpringIndices, springIndices, numSprings*2*sizeof(uint32_t));
	memcpy(s->mSpringLengths, springLengths, numSprings*sizeof(float));
	
	s->mNumSprings = numSprings;
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

void grainGetDensities(GrainSystem* s, float* r)
{
	memcpy(r, &s->mDensity[0], sizeof(float)*s->mNumGrains);
}

void grainGetVorticities(GrainSystem* s, float* r)
{
	memcpy(r, &s->mVorticity[0], sizeof(float)*s->mNumGrains);
}

void grainGetMass(GrainSystem* s, float* r)
{
	memcpy(r, &s->mDensity[0], sizeof(float)*s->mNumGrains);
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

#endif
