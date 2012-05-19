#pragma once

#include "Types.h"
#include "Log.h"

#include <vector>
#include <string>

// system functions
double GetSeconds();
void Sleep(double seconds);

// helper function to get exe path
tstring GetExePath();
tstring GetWorkingDirectory();

// shows a file open dialog
tstring FileOpenDialog(char *filter = "All Files (*.*)\0*.*\0");

// pulls out an option in the form option=value, must have no spaces
template <typename T>
bool GetCmdLineArg(const tchar* arg, T& out, int argc, char* argv[])
{
	// iterate over all the arguments
	for (int i=0; i < argc; ++i)
	{
		const tchar* s1 = arg;
		const tchar* s2 = argv[i];

		while (*s1 && *s2 && *s1 == *s2)
		{
			++s1;
			++s2;
		}
		
		// we've found the flag we're looking for
		if (*s1 == 0 && *s2 == '=')
		{
			++s2;

			// build a string stream and output			
			std::istringstream is(s2);
			if (is >> out)
			{
				return true;
			}
			else
				return false;
		}
	}

	return false;
}
// return the full path to a file
tstring ExpandPath(const tchar* path);
// takes a full file path and returns just the folder (with trailing slash)
tstring StripFilename(const tchar* path);
// strips the path from a file name
tstring StripPath(const tchar* path);
// strips the extension from a path
tstring StripExtension(const tchar* path);
// returns the file extension (excluding period)
tstring GetExtension(const tchar* path);
// normalize path
tstring NormalizePath(const tchar* path);

tchar* AsciiToUnicode(const char* in, tchar* out, int count);
char* UnicodeToAscii(const tchar* in, char* out, int count);

// loads a file to a text string
tstring LoadFileToString(const char* filename);
// loads a file to a binary buffer (free using delete[])
byte* LoadFileToBuffer(const char* filename, uint32* sizeRead=NULL);
// save whole string to a file
bool SaveStringToFile(const char* filename, const char* s);

bool FileMove(const char* src, const char* dest);
bool FileScan(const char* pattern, std::vector<std::string>& files);

// file system stuff
const uint kMaxPathLength = 2048;

#ifdef WIN32

// defined these explicitly because don't want to include windowsx.h
#define GET_WPARAM(wp, lp)                      (wp)
#define GET_LPARAM(wp, lp)                      (lp)

#define GET_X_LPARAM(lp)                        ((int)(short)LOWORD(lp))
#define GET_Y_LPARAM(lp)                        ((int)(short)HIWORD(lp))

#define vsnprintf _vsnprintf
#define snprintf _snprintf
#define vsnwprintf _vsnwprintf

#if _MSC_VER >= 1400 //vc8.0 use new secure
#define snwprintf _snwprintf_s
#else
#define snwprintf _snwprintf
#endif // _MSC_VER

#endif // WIN32

#if PLATFORM_IOS
inline tstring ExpandPath(const tchar* p)
{
	NSString *imagePath = [NSString stringWithUTF8String:p];
	NSString *fullPath = [[NSBundle mainBundle] pathForResource:[imagePath lastPathComponent] ofType:nil inDirectory:[imagePath stringByDeletingLastPathComponent]];
	
	if (fullPath)
	{
		tstring s = [fullPath cStringUsingEncoding:1];		
		return s;
	}
	else 
	{
		Log::Info << "Failed to map path for : " << p << std::endl;
		return tstring("");
	}
}
inline tstring GetTempDirectory()
{
	NSString* tmp = NSTemporaryDirectory();
	tstring s = [tmp cStringUsingEncoding:1];
	
	return s;
}

inline tstring DataPath(const tchar* p)
{
	return ExpandPath((tstring("DataCooked/") + p).c_str());
}

#else

inline tstring ExpandPath(const tchar* p)
{
	return p;
}

inline tstring DataPath(const tchar* p)
{
	return ExpandPath((tstring("Data/") + p).c_str());

}

#endif
