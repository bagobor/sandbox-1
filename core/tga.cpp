#include "tga.h"
#include "core.h"
#include "types.h"
#include "log.h"

#include <stdio.h>

using namespace std;

struct TgaHeader
{
    uint8_t  identsize;          // size of ID field that follows 18 uint8_t header (0 usually)
    uint8_t  colourmaptype;      // type of colour map 0=none, 1=has palette
    uint8_t  imagetype;          // type of image 0=none,1=indexed,2=rgb,3=grey,+8=rle packed

    uint16_t colourmapstart;     // first colour map entry in palette
    uint16_t colourmaplength;    // number of colours in palette
    uint8_t  colourmapbits;      // number of bits per palette entry 15,16,24,32

    uint16_t xstart;             // image x origin
    uint16_t ystart;             // image y origin
    uint16_t width;              // image width in pixels
	uint16_t height;             // image height in pixels
    uint8_t  bits;               // image bits per pixel 8,16,24,32
    uint8_t  descriptor;         // image descriptor bits (vh flip bits)
    
    // pixel data follows header  
};


bool TgaSave(const char* filename, const TgaImage& image)
{
	TgaHeader header;

	header.identsize = 0;
	header.colourmaptype = 0;
	header.imagetype = 2;
	header.colourmapstart = 0;
	header.colourmaplength = 0;
	header.colourmapbits = 0;
	header.xstart = 0;
	header.ystart = 0;
	header.width = image.m_width;
	header.height = image.m_height;
	header.bits = 32;
	header.descriptor = 0;//uint16((1<<3)|(1<<5));

	FILE* f = fopen(filename, "wb");
	if (f)
	{
		fwrite(&header.identsize, 1, sizeof(header.identsize), f);
		fwrite(&header.colourmaptype, 1, sizeof(header.colourmaptype), f);
		fwrite(&header.imagetype, 1, sizeof(header.imagetype), f);
		fwrite(&header.colourmapstart, 1, sizeof(header.colourmapstart), f);
		fwrite(&header.colourmaplength, 1, sizeof(header.colourmaplength), f);
		fwrite(&header.colourmapbits, 1, sizeof(header.colourmapbits), f);
		fwrite(&header.xstart, 1, sizeof(header.xstart), f);
		fwrite(&header.ystart, 1, sizeof(header.ystart), f);
		fwrite(&header.width, 1, sizeof(header.width), f);
		fwrite(&header.height, 1, sizeof(header.height), f);
		fwrite(&header.bits, 1, sizeof(header.bits), f);
		fwrite(&header.descriptor, 1, sizeof(header.descriptor), f);

		union texel
		{
			uint32_t u32;
			uint8_t u8[4];
		};
		
		texel* t = (texel*)image.m_data;
		for (int i=0; i < image.m_width*image.m_height; ++i)
		{
			//texel tmp = t[i];
			//swap(tmp.u8[0], tmp.u8[2]);
			swap(t[i].u8[0], t[i].u8[2]);
		}
			 
		fwrite(t, 1, image.m_width*image.m_height*4, f);
		fclose(f); 

		return true;
	}	
	else
	{
		return false;
	}
}


bool TgaLoad(const char* filename, TgaImage& image)
{
	if (!filename)
		return false;

	FILE* aTGAFile = fopen(filename, "rb");
	if (aTGAFile == NULL)
	{
		printf("Texture: could not open %s for reading.\n", filename);
		return NULL;
	}

	char aHeaderIDLen;
	fread(&aHeaderIDLen, sizeof(uint8_t), 1, aTGAFile);

	char aColorMapType;
	fread(&aColorMapType, sizeof(uint8_t), 1, aTGAFile);
	
	char anImageType;
	fread(&anImageType, sizeof(uint8_t), 1, aTGAFile);

	short aFirstEntryIdx;
	fread(&aFirstEntryIdx, sizeof(uint16_t), 1, aTGAFile);

	short aColorMapLen;
	fread(&aColorMapLen, sizeof(uint16_t), 1, aTGAFile);

	char aColorMapEntrySize;
	fread(&aColorMapEntrySize, sizeof(uint8_t), 1, aTGAFile);	

	short anXOrigin;
	fread(&anXOrigin, sizeof(uint16_t), 1, aTGAFile);

	short aYOrigin;
	fread(&aYOrigin, sizeof(uint16_t), 1, aTGAFile);

	short anImageWidth;
	fread(&anImageWidth, sizeof(uint16_t), 1, aTGAFile);	

	short anImageHeight;
	fread(&anImageHeight, sizeof(uint16_t), 1, aTGAFile);	

	char aBitCount = 32;
	fread(&aBitCount, sizeof(uint8_t), 1, aTGAFile);	

	char anImageDescriptor;// = 8 | (1<<5);
	fread((char*)&anImageDescriptor, sizeof(uint8_t), 1, aTGAFile);
	
	// total is the number of bytes we'll have to read
	uint8_t numComponents = aBitCount / 8;
	uint32_t numTexels = anImageWidth * anImageHeight;

	// allocate memory for image pixels
	image.m_width = anImageWidth;
	image.m_height = anImageHeight;
	image.m_data = new uint32_t[numTexels];

	// load the image pixels
	for (uint32_t i=0; i < numTexels; ++i)
	{
		union texel
		{
			uint32_t u32;
			uint8_t u8[4];
		};

		texel t;
		t.u32 = 0;

		if (!fread(&t.u32, numComponents, 1, aTGAFile))
		{
			printf("Texture: file not fully read, may be corrupt (%s)\n", filename);
		}

		// stores it as BGR(A) so we'll have to swap R and B.
		swap(t.u8[0], t.u8[2]);


		image.m_data[i] = t.u32;
	}

	// if bit 5 of the descriptor is set then the image is flipped vertically so we fix it up
	if (anImageDescriptor & (1 << 5))
	{

		// swap all the rows
		int rowSize = image.m_width*4;	

		uint8_t* buf = new uint8_t[image.m_width*4];
		uint8_t* start = (uint8_t*)image.m_data;
		uint8_t* end = &((uint8_t*)image.m_data)[rowSize*(image.m_height-1)];
		
		while (start < end)
		{
			memcpy(buf, end, rowSize);
			memcpy(end, start, rowSize);
			memcpy(start, buf, rowSize);

			start += rowSize;
			end -= rowSize;
		}

		delete[] buf;
	}

	fclose(aTGAFile);
	
	return true;
}
