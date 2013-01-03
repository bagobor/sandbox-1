#include "pfm.h"

#include <cassert>
#include <stdio.h>
#include <string.h>
#include <algorithm>

namespace
{
	// RAII wrapper to handle file pointer clean up
	struct FilePointer
	{
		FilePointer(FILE* ptr) : p(ptr) {}
		~FilePointer() { fclose(p); }

		operator FILE*() { return p; }

		FILE* p;
	};
}

bool PfmLoad(const char* filename, PfmImage& image)
{
	FilePointer f = fopen(filename, "rb");
	if (!f)
		return false;
	
	const uint32_t kBufSize = 1024;
	char buffer[kBufSize];
	
	if (!fgets(buffer, kBufSize, f))
		return false;
	
	if (strcmp(buffer, "PF\n") != 0)
		return false;
	
	if (!fgets(buffer, kBufSize, f))
		return false;
	
	sscanf(buffer, "%d %d", &image.m_width, &image.m_height);
	
	if (!fgets(buffer, kBufSize, f))
		return false;
	
	sscanf(buffer, "%f", &image.m_maxDepth);
	
	uint32_t dataStart = ftell(f);
	fseek(f, 0, SEEK_END);
	uint32_t dataEnd = ftell(f);
	fseek(f, dataStart, SEEK_SET);
	
	uint32_t dataSize = dataEnd-dataStart;
	assert((dataSize&0x3) == 0);
	
	// determine if the rest of the image is RGB or scalar
	image.m_data = new float[dataSize/4];
	
	if (fread(image.m_data, dataSize, 1, f) != 1)
		return false;
	
	return true;
}

void PfmSave(const char* filename, const PfmImage& image)
{
	FILE* f = fopen(filename, "wb");
	if (!f)
		return;

	fprintf(f, "PF\n");
	fprintf(f, "%d %d\n", image.m_width, image.m_height);
	fprintf(f, "%f\n", *std::max_element(image.m_data, image.m_data+(image.m_width*image.m_height)));

	fwrite(image.m_data, image.m_width*image.m_height*sizeof(float), 1, f);
}





