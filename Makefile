CTAGS  = ctags --fields=+l --c-kinds=+p --c++-kinds=+p 

FILES  = core/*.h external/glut/glut.h /Developer/SDKs/MacOSX10.6.sdk/System/Library/Frameworks/OpenGL.framework/Versions/A/Headers/gl.h 
tags: $(FILES) Makefile
	$(CTAGS) $(FILES) 
	
