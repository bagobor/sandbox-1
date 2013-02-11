#include "core/aabbtree.h"
#include "core/mesh.h"

void Voxelize(const Mesh& mesh, uint32_t width, uint32_t height, uint32_t depth, uint32_t* volume, Vec3 minExtents, Vec3 maxExtents)
{
	memset(volume, 0, sizeof(uint32_t)*width*height*depth);

	// build an aabb tree of the mesh
	AABBTree tree(&mesh.m_positions[0], mesh.m_positions.size(), &mesh.m_indices[0], mesh.m_indices.size()/3); 

	// parity count method, single pass
	const Vec3 extents(maxExtents-minExtents);
	const Vec3 delta(extents.x/width, extents.y/height, extents.z/depth);
	const Vec3 offset(0.5f/width, 0.5f/height, 0.5f/depth);
	
	for (uint32_t x=0; x < width; ++x)
	{
		for (uint32_t y=0; y < height; ++y)
		{
			uint32_t z = 0; 
			bool inside = false;

			while (z < depth)
			{
				// calculate ray start
				const Vec3 rayStart = minExtents + Vec3(x*delta.x + offset.x, y*delta.y + offset.y, z*delta.z);
				const Vec3 rayDir = Vec3(0.0f, 0.0f, 1.0f);

				float t, u, v, w, s;
				uint32_t tri;

				if (tree.TraceRay(Point3(rayStart), rayDir, t, u, v, w, s, tri))
				{
					// calculate cell in which intersection occurred
					const float zpos = rayStart.z + t*rayDir.z;
					const float zhit = ceilf((zpos-minExtents.z)/delta.z);
					uint32_t zend = std::min(uint32_t(zhit), depth);

					// must be true for termination
					assert(zend >= z);
					if (zend == z && zend < depth-1)
						zend++;

					if (inside)
					{
						// march along column setting bits 
						for (uint32_t k=z; k < zend; ++k)
							volume[k*width*height + y*width + x] = uint32_t(-1);
					}

					inside = !inside;

					z = zend;
				}
				else
					break;
			}
		}
	}	
}
