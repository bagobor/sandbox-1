#pragma once

#include "types.h"

struct TgaImage
{
	uint16 m_width;
	uint16 m_height;

	// pixels are always assumed to be 32 bit
	uint32* m_data;
};

bool TgaSave(const char* filename, const TgaImage& image);
bool TgaLoad(const char* filename, TgaImage& image);
