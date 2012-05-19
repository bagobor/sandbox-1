CTAGS  = ctags --fields=+l --c-kinds=+p --c++-kinds=+p 

FILES  = Core/*.h /Developer/SDKs/MacOSX10.6.sdk/System/Library/Frameworks/OpenGL.framework/Versions/A/Headers/gl.h External/Glut/glut.h

tags: $(FILES) Makefile
	$(CTAGS) $(FILES) 
	
