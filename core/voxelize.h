#pragma once

struct Mesh;

// voxelizes a mesh using a single pass parity algorithm
void Voxelize(const Mesh& mesh, uint32_t width, uint32_t height, uint32_t depth, uint32_t* volume);

// generate the signed distance field for a volume, voxels > 0 assumed to be non-empty
void DistanceField(uint32_t width, uint32_t height, uint32_t depth, const uint32_t* volume, float* sdf);