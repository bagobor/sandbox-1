#include "Assertion.h"

#ifndef NDEBUG

#if WIN32

#include <windows.h>
#include <stdio.h>
#include <tchar.h>

bool CustomAssert(const tchar *exp, const tchar *msg, int line, const char *file, bool& ignoreAlways)
{
	// find top-level window of our app if possible
	HWND hwnd = GetActiveWindow();
	if (hwnd)
		hwnd = GetLastActivePopup(hwnd);

	// ok fine i'll use the 'secure' version
	tchar buf[512];
	_sntprintf_s(buf, sizeof(buf), _TRUNCATE, _TS("A program error has occurred:\n\n")
	                                         _TS("File: %s\n")
	                                         _TS("Line: %d\n")
	                                         _TS("Expression: %s\n")
	                                         _TS("Message: %s\n\n")
	                                         _TS("Would you like to debug? Click Cancel to ignore this error from now on."),
	                                         file, line, exp, (msg ? msg : _TS("None specified")));

	// display message box
	int res = MessageBox(hwnd, buf, _TS("Assertion failed"), MB_YESNOCANCEL | MB_ICONERROR | MB_SETFOREGROUND | MB_TASKMODAL);
	if (res == IDCANCEL)
		ignoreAlways = true;
	
	// return true if they asked to debug
	return (res == IDYES);
}

#else

#include "Log.h"

bool CustomAssert(const tchar *exp, const tchar *msg, int line, const char *file, bool& ignoreAlways)
{
	Log::Error << exp << ", " << msg << ":" << line << ", " << file << std::endl;
	return false;
}

#endif // WIN32

#endif	// NDEBUG

