#pragma once

#include "Core.h"
#include "Maths.h"
#include "Tga.h"
#include "Memory.h"

class Image
{
public:

	struct Colour32
	{
		Colour32() : b(0), g(0), r(0), a(255) {}
		Colour32(byte inR, byte inG, byte inB, byte inA) : b(inB), g(inG), r(inR), a(inA)
		{
		}

		// store in BGRA format for ease of TGA export
		byte b;
		byte g;
		byte r;
		byte a;
	};

	Image(uint32 width, uint32 height) : m_width(width), m_height(height)
	{
		m_pixels = (Colour*)AlignedMalloc(sizeof(Colour)*width*height, 16);
		memset(m_pixels, 0, sizeof(Colour)*width*height);
	}
	

	~Image()
	{
		AlignedFree(m_pixels);
	}

	void SetPixel(uint32 x, uint32 y, const Colour& c)
	{
		m_pixels[y*m_width + x] = c;
	}

	const Colour& GetPixel(uint32 x, uint32 y) { return m_pixels[y*m_width + x]; }

	void ToneMap()
	{
		return;
		
		const float kYWeight[3] = { 0.212671f, 0.715160f, 0.072169f };
		
		
		// compute world adaption luminance
		for (uint32 i=0; i < m_width*m_height; ++i)
		{
			// calculate pixel luminance
			float y = (kYWeight[0]*m_pixels[i].r + kYWeight[1]*m_pixels[i].g + kYWeight[2]*m_pixels[i].b);
			
			float scale = y / (1.0f + y);

			m_pixels[i] *= scale;
		}
		
		
		/*
		float maxY = 0.0f;

		for (uint32 i=0; i < m_width*m_height; ++i)
		{
			float y = (kYWeight[0]*m_pixels[i].r + kYWeight[1]*m_pixels[i].g + kYWeight[2]*m_pixels[i].b);
			if (y > maxY)
				maxY = y;
		}

		float s = 1.0f / maxY;

		for (uint32 i=0; i < m_width*m_height; ++i)
		{
			m_pixels[i] *= s;			
		}
		*/
		
	}

#ifndef PLATFORM_SPU

	void SaveToFile(const char* filename)
	{
		ToneMap();

		// convert to 8 bit
		Colour32* p = new Colour32[m_width*m_height];
		
		for (uint32 i=0; i < m_height; ++i)
		{
			for (uint32 j=0; j < m_width; ++j)
			{	
				// clamp between 0.0-1.0
				uint32 index = i*m_width+j;
				
				m_pixels[index].r = Clamp(m_pixels[index].r, 0.0f, 1.0f);
				m_pixels[index].g = Clamp(m_pixels[index].g, 0.0f, 1.0f);
				m_pixels[index].b = Clamp(m_pixels[index].b, 0.0f, 1.0f); 
				
				m_pixels[index] = LinearToSrgb(m_pixels[index]);

				p[index].r = byte(m_pixels[index].r*255.0f);
				p[index].g = byte(m_pixels[index].g*255.0f);
				p[index].b = byte(m_pixels[index].b*255.0f);
				p[index].a = 255;
			}
		}

		TgaImage image;
		image.m_width = (uint16)m_width;
		image.m_height = (uint16)m_height;
		image.m_data = (uint32*)p;
		
		TgaSave(filename, image);

		delete[] p;
	}

#endif

	uint32 m_width;
	uint32 m_height;

	Colour* m_pixels;
};