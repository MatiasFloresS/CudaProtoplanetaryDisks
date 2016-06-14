
CFLAGS = -c -w
LDFLAGS = -lcuda -lcudart
INC_DIRS = -Iinclude

main: build/main.o build/Viscosity.o build/kernels.o build/readfiles.o build/SourceEuler.o
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

build/kernels.o: src/kernels.cu
	@ echo "Building kernels"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/kernels.cu -o build/kernels.o

build/readfiles.o: src/readfiles.cu
	@ echo "Building readfiles"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/readfiles.cu -o build/readfiles.o

build/SourceEuler.o: src/SourceEuler.cu
	@ echo "Builing SourceEuler"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/SourceEuler.cu -o build/SourceEuler.o
	
clean:
	@ clear
	@ echo "Cleaning folders..."
	@ rm -rf build/*
	@ rm -rf bin/*
