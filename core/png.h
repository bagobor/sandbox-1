#pragma once

#include "types.h"

struct PngImage
{
	uint16 m_width;
	uint16 m_height;

	// pixels are always assumed to be 32 bit
	uint32* m_data;
};

bool PngLoad(const char* filename, PngImage& image);
