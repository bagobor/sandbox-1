#include "types.h"

struct PfmImage
{
	uint32_t m_width;
	uint32_t m_height;
	float m_maxDepth;

	float* m_data;
};

bool PfmLoad(const char* filename, PfmImage& image);
void PfmSave(const char* filename, const PfmImage& image);
