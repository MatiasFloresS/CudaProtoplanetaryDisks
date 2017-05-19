
CFLAGS = -c -w
CUFFTFLAG = -lcufft
LDFLAGS = -lcuda -lcudart
INC_DIRS = -Iinclude

SMS ?= 30 35 37 50 52

ifeq ($(SMS),)
$(info >>> WARNING - no SM architectures have been specified <<<)
endif

ifeq ($(ARCHFLAG),)
# Generate SASS code for each SM architecture listed in $(SMS)
$(foreach sm,$(SMS),$(eval ARCHFLAG += -gencode arch=compute_$(sm),code=sm_$(sm)))

# Generate PTX code from the highest SM architecture in $(SMS) to guarantee forward-compatibility
HIGHEST_SM := $(lastword $(sort $(SMS)))
ifneq ($(HIGHEST_SM),)
ARCHFLAG += -gencode arch=compute_$(HIGHEST_SM),code=compute_$(HIGHEST_SM)
endif
endif


main: build/Main.o build/Viscosity.o build/Interpret.o build/SourceEuler.o build/Psys.o build/Pframeforce.o \
	build/Theo.o build/Init.o build/SideEuler.o build/Output.o build/Force.o build/TransportEuler.o build/Planet.o \
	build/RungeKutta.o build/SgMain.o build/LowTasks.o build/Kernels.o
	@ echo "Linking"
	@ mkdir -p bin
	@ nvcc build/*.o -o bin/fargoGPU $(LDFLAGS) $(CUFFTFLAG) $(ARCHFLAG)

build/Main.o: src/Main.cu
	@ echo "Building Main"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Main.cu -o build/Main.o $(LDFLAGS) $(CUFFTFLAG) $(ARCHFLAG)

build/Viscosity.o: src/Viscosity.cu
	@ echo "Building Viscosity"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Viscosity.cu -o build/Viscosity.o $(LDFLAGS) $(ARCHFLAG)

build/Kernels.o: src/Kernels.cu
	@ echo "Building Kernels"
	@ nvcc $(CFLAGS) $(INC_DIRS)  src/Kernels.cu -o build/Kernels.o $(LDFLAGS) $(ARCHFLAG) $(CUFFTFLAG)

build/Interpret.o: src/Interpret.cu
	@ echo "Building Interpret"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Interpret.cu -o build/Interpret.o $(LDFLAGS) $(ARCHFLAG)

build/SourceEuler.o: src/SourceEuler.cu
	@ echo "Building SourceEuler"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/SourceEuler.cu -o build/SourceEuler.o $(LDFLAGS) $(ARCHFLAG)

build/Psys.o: src/Psys.cu
	@ echo "Building Psys"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Psys.cu -o build/Psys.o $(LDFLAGS) $(ARCHFLAG)

build/Pframeforce.o: src/Pframeforce.cu
	@ echo "Building Pframeforce"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Pframeforce.cu -o build/Pframeforce.o $(LDFLAGS) $(ARCHFLAG)

build/Theo.o: src/Theo.cu
	@ echo "Building Theo"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Theo.cu -o build/Theo.o $(LDFLAGS) $(ARCHFLAG)

build/Init.o: src/Init.cu
	@ echo "Building Init"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Init.cu -o build/Init.o $(LDFLAGS) $(ARCHFLAG)

build/SideEuler.o: src/SideEuler.cu
	@ echo "Building SideEuler"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/SideEuler.cu -o build/SideEuler.o $(LDFLAGS) $(ARCHFLAG)

build/Output.o: src/Output.cu
	@ echo "Building Output"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Output.cu -o build/Output.o $(LDFLAGS) $(ARCHFLAG)

build/Force.o: src/Force.cu
	@ echo "Building Force"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Force.cu -o build/Force.o $(LDFLAGS) $(ARCHFLAG)

build/TransportEuler.o: src/TransportEuler.cu
	@ echo "Building TransportEuler"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/TransportEuler.cu -o build/TransportEuler.o $(LDFLAGS) $(ARCHFLAG)

build/Planet.o: src/Planet.cu
	@ echo "Building Planet"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Planet.cu -o build/Planet.o $(LDFLAGS) $(ARCHFLAG)

build/RungeKutta.o: src/RungeKutta.cu
	@ echo "Building RungeKutta"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/RungeKutta.cu -o build/RungeKutta.o $(LDFLAGS) $(ARCHFLAG)

build/SgMain.o: src/SgMain.cu
	@ echo "Building SgMain"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/SgMain.cu -o build/SgMain.o $(LDFLAGS) $(ARCHFLAG)

build/LowTasks.o: src/LowTasks.cu
	@ echo "Building LowTasks"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/LowTasks.cu -o build/LowTasks.o $(LDFLAGS) $(ARCHFLAG)

clean:
	@ clear
	@ echo "Cleaning folders..."
	@ rm -rf build/*
	@ rm -rf bin/*
	@ rm -rf out/*

cleanout:
	@ clear
	@ echo "Clening output folder..."
	@ rm -rf out/*
