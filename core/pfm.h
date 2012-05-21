/*
 *  Pfm.h
 *  Surfel
 *
 *  Created by Miles Macklin on 16/04/11.
 *  Copyright 2011 None. All rights reserved.
 *
 */

#include "types.h"

struct PfmImage
{
	uint32_t m_width;
	uint32_t m_height;
	float m_maxDepth;
	
	// pixels are always assumed to be 32 bit
	float* m_data;
};

bool PfmLoad(const char* filename, PfmImage& image);