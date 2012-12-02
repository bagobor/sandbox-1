#pragma once

#ifdef CUDA
#include <cutil_math.h>
#include <cutil_inline.h>    // includes cuda.h and cuda_runtime_api.h
#else
typedef Vec2 float2;
typedef Vec3 float3;
#endif


struct GrainSystem;

//C:/p4/physx/../tools/sdk/CUDA/4.0/4.0.17/bin32\nvcc.exe -m32 --ptxas-options=-v -use_fast_math -ftz=true -prec-div=false -prec-sqrt=false -gencode=arch=compute_11,code=sm_11 -gencode=arch=compute_12,code=sm_12 -gencode=arch=compute_20,code=sm_20 --compiler-bindir="$(VCInstallDir)bin" --compile -D_DEBUG -DWIN32 -D_CONSOLE -D_WIN32_WINNT=0x0500 --compiler-options=/EHsc,/W3,/nologo,/Ot,/Ox,/Zi,/MTd,/Fd./Win32/PhysXGpu/debug/CUDA_src/PackParticleShapes.obj.pdb -IC:/p4/physx/PhysXSDK/3.2/trunk/Include/foundation -IC:/p4/physx/PhysXSDK/3.2/trunk/Source/foundation/include -I../../../Source/PhysXGpu/src/common -I../../../Source/LowLevel/common/include/math -I../../../Include/geometry -I../../../Source/GeomUtils/headers -I../../../Source/GeomUtils/src -I../../../Source/LowLevel/API/include -I../../../Include/common  -I../../../Source/Common/src -I../../../Include -I../../../pxtask/CUDA -o ./Win32/PhysXGpu/debug/CUDA_src/PackParticleShapes.obj ..\..\PhysXGpu\src\CUDA\PackParticleShapes.cu

struct GrainParams
{
	float2 mGravity;
	float mDamp;
	
	float mBaumgarte;
	float mFriction;
	float mRestitution;
	float mOverlap;

	// fluid params
	float mMass;
	float mRestDensity;

	float3 mPlanes[8];
	int mNumPlanes;
};

struct GrainTimers
{
	GrainTimers() 
		: mCreateCellIndices(0.0f)
		, mSortCellIndices(0.0f)
		, mCreateGrid(0.0f)
		, mCollide(0.0f)
		, mIntegrate(0.0f)
		, mReorder(0.0f)
	{
	}

	float mCreateCellIndices;
	float mSortCellIndices;
	float mCreateGrid;
	float mCollide;
	float mIntegrate;
	float mReorder;
};

GrainSystem* grainCreateSystem(int numGrains);
void grainDestroySystem(GrainSystem* s);

void grainSetSprings(GrainSystem* s, const uint32_t* indices, const float* restLengths, uint32_t numSprings);

void grainSetPositions(GrainSystem* s, float* p, int n);
void grainSetVelocities(GrainSystem* s, float* v, int n);

void grainGetPositions(GrainSystem* s, float* p);
void grainGetVelocities(GrainSystem* s, float* v);

void grainSetRadii(GrainSystem* s, float* r);
void grainGetRadii(GrainSystem* s, float r);

void grainGetDensities(GrainSystem* s, float* r);
void grainGetVorticities(GrainSystem* s, float* r);

void grainGetMass(GrainSystem* s, float* r);

void grainSetParams(GrainSystem* s, GrainParams* params);

void grainUpdateSystem(GrainSystem* s, float dt, int iterations, GrainTimers* timers);
