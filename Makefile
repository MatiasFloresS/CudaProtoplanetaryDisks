
CFLAGS = -c -w
LDFLAGS = -lcuda -lcudart
INC_DIRS = -Iinclude

main: build/main.o build/Viscosity.o build/kernels.o build/readfiles.o build/SourceEuler.o build/Psys.o build/Pframeforce.o \
	build/Theo.o build/Init.o build/SideEuler.o
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
	@ echo "Building SourceEuler"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/SourceEuler.cu -o build/SourceEuler.o

build/Psys.o: src/Psys.cu
	@ echo "Building Psys"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/Psys.cu -o build/Psys.o

build/Pframeforce.o: src/Pframeforce.cu
	@ echo "Building Pframeforce"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/Pframeforce.cu -o build/Pframeforce.o

build/Theo.o: src/Theo.cu
	@ echo "Building Theo"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/Theo.cu -o build/Theo.o

build/Init.o: src/Init.cu
	@ echo "Building Init"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/Init.cu -o build/Init.o

build/SideEuler.o: src/SideEuler.cu
	@ echo "Building SideEuler"
	@ nvcc $(CFLAGS) $(INC_DIRS) $(LDFLAGS) src/SideEuler.cu -o build/SideEuler.o


clean:
	@ clear
	@ echo "Cleaning folders..."
	@ rm -rf build/*
	@ rm -rf bin/*
