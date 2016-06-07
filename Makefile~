
CFLAGS = -c -w
LDFLAGS = -lcuda -lcudart
INC_DIRS = -Iinclude

main: build/main.o build/Viscosity.o
	@ echo "Linking"
	@ mkdir -p bin
	@ nvcc $(LDFLAGS) build/*.o -o bin/fargoGPU 

build/main.o: src/main.cu
	@ echo "Building Main"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/main.cu -o build/main.o 

build/Viscosity.o: src/Viscosity.cu
	@ echo "Building Viscosity"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/Viscosity.cu -o build/Viscosity.o 
