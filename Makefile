APP=forcetracker

CC=g++
CPPFLAGS=-I/usr/local/Cellar/opencv/2.3.1a/include
OPENCV_LIBS= \
	-lopencv_core \
	-lopencv_highgui \
	-lopencv_objdetect \
	-lopencv_imgproc

LDLIBS=-L/usr/local/Cellar/opencv/2.3.1a/lib $(OPENCV_LIBS)

SRCS=$(wildcard *.cpp)
OBJS=$(patsubst %.cpp,%.o,$(SRCS))


all: $(OBJS)
	$(CC) $(CPPFLAGS) $(LDLIBS) $< -o $(APP)

run: all
	./$(APP)

clean:
	rm -fr $(OBJS) $(APP)