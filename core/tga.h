#pragma once

#include "types.h"

#include <algorithm>

struct TgaImage
{
	~TgaImage() { delete m_data; }

	uint32_t SampleClamp(int x, int y) const
	{
		uint32_t ix = std::min(std::max(0, x), int(m_width));
		uint32_t iy = std::min(std::max(0, y), int(m_height));

		return m_data[iy*m_width + ix];
	}

	uint16_t m_width;
	uint16_t m_height;

	// pixels are always assumed to be 32 bit
	uint32_t* m_data;
};

bool TgaSave(const char* filename, const TgaImage& image);
bool TgaLoad(const char* filename, TgaImage& image);
