#pragma once

#include "types.h"

struct TgaImage
{
	~TgaImage() { delete m_data; }

	uint16_t m_width;
	uint16_t m_height;

	// pixels are always assumed to be 32 bit
	uint32_t* m_data;
};

bool TgaSave(const char* filename, const TgaImage& image);
bool TgaLoad(const char* filename, TgaImage& image);
