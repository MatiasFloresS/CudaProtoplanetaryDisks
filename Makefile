
CFLAGS = -c -w
LDFLAGS = -lcuda -lcudart
INC_DIRS = -Iinclude

SMS ?= 20 30 35 37 50 52

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



main: build/main.o build/Viscosity.o build/kernels.o build/readfiles.o build/SourceEuler.o build/Psys.o build/Pframeforce.o \
	build/Theo.o build/Init.o build/SideEuler.o build/Output.o build/Force.o
	@ echo "Linking"
	@ mkdir -p bin
	@ nvcc build/*.o -o bin/fargoGPU $(LDFLAGS) $(ARCHFLAG)

build/main.o: src/main.cu
	@ echo "Building Main"
	@ mkdir -p build
	@ nvcc $(CFLAGS) $(INC_DIRS) src/main.cu -o build/main.o $(LDFLAGS) $(ARCHFLAG)

build/Viscosity.o: src/Viscosity.cu
	@ echo "Building Viscosity"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/Viscosity.cu -o build/Viscosity.o $(LDFLAGS) $(ARCHFLAG)

build/kernels.o: src/kernels.cu
	@ echo "Building kernels"
	@ nvcc $(CFLAGS) $(INC_DIRS)  src/kernels.cu -o build/kernels.o $(LDFLAGS) $(ARCHFLAG)

build/readfiles.o: src/readfiles.cu
	@ echo "Building readfiles"
	@ nvcc $(CFLAGS) $(INC_DIRS) src/readfiles.cu -o build/readfiles.o $(LDFLAGS) $(ARCHFLAG)

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

