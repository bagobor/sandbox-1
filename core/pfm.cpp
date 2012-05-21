/*
 *  Pfm.cpp
 *  Surfel
 *
 *  Created by Miles Macklin on 16/04/11.
 *  Copyright 2011 None. All rights reserved.
 *
 */

#include "pfm.h"
#include <cassert>

bool PfmLoad(const char* filename, PfmImage& image)
{
	FILE* f = fopen(filename, "rb");
	if (!f)
		return false;
		
	const uint32 kBufSize = 1024;
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
	
	uint32 dataStart = ftell(f);
	fseek(f, 0, SEEK_END);
	uint32 dataEnd = ftell(f);
	fseek(f, dataStart, SEEK_SET);
	
	uint32 dataSize = dataEnd-dataStart;
	assert((dataSize&0x4) == 0);
	
	// determine if the rest of the image is RGB or scalar
	image.m_data = new float[dataSize/4];
	
	if (fread(image.m_data, dataSize, 1, f) != 1)
		return false;
	
	return true;
}
