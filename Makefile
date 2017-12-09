all: ind.cpp
	g++ ind.cpp -o ind -lX11 -lpthread -I. -Isiftpp -O3 siftpp/sift.cpp -std=c++11 `pkg-config opencv --cflags --libs` 

clean:
	rm ind
