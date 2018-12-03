#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <device_launch_parameters.h>
#include <cuda_runtime.h>
#include <curand.h>

long double goldenNum = powl((long double)(0.5), (long double)(16)); // (1/2)^16
void query_device();

int main(int argc, char **argv)
{
	printf("---------------------------------------------\n");
	query_device();
	printf("\n\n---------------------------------------------\n");
	printf("\tSets of {16} flips\n");
	printf("Minimum number that will run: 1024\n");
	printf("How many multiples of 1024? : ");
	unsigned long long int input = 1;
	scanf("%lld", &input);
	printf("---------------------------------------------\n");
	printf("--- Flipping [%lld] sets of 16 coins ---\n", (input*1024));
	printf("\t( \"%lldM sets\" or \"%lldM flips\" )\n", ((input * 1024)/1000000), ((input * 1024 * 16) / 1000000));
	printf("---------------------------------------------\n\n");
	// -------------------------------------------------------
	// TIMER
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// START  -------------------------------------------------
	cudaEventRecord(start);
	// -------------------------------------------------------
	// HHT HT HHHT HHHT HT H
  // H = 1
	// T = 0
	int succ = 0, fail = 0;
	const int flips[] = { 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1 };
	// -------------------------------------------------------
	// Mersenne Twister number generator
	// Each output element is a 32-bit unsigned int where all bits are random
	curandGenerator_t genGPU;
	curandCreateGenerator(&genGPU, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(genGPU, clock());
	// "Hence the most efficient use of MTGP32 is to generate a multiple of 16384 samples."
	// -------------------------------------------------------
	const int x = 16, y = 1024, n = x * y; // 16384
	unsigned int GPU[n], *d_GPU;
	cudaMalloc(&d_GPU, n * sizeof(unsigned int));
	// -------------------------------------------------------
	printf("Now THIS is pod racing!\n");
	int offset;
	for (int z = 1; z <= input; z++) {
		// generate numbers
		curandGenerate(genGPU, d_GPU, n);
		// cpy back to host
		cudaMemcpy(GPU, d_GPU, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		// check for matches
		for (int i = 0; i < y; ++i) {	  // ROW (y)
			for (int j = 0; j < x; ++j) { // COL (x)
				offset = i + j;
				GPU[offset] = (GPU[offset] % 2);
				//printf("%u", GPU[offset]);
				//printf("%d:%d ", GPU[offset], flips[j]);
				if (GPU[offset] == flips[j]) {
					if (j == 15) { // if all match
						succ++;
						break;
					}
				}
				else {
					fail++;
					break;
				}
			}
			//printf("\n");
		}
		// do it again, lol
	}
	// STOP  -------------------------------------------------
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float seconds = (milliseconds / 1000);
	// -------------------------------------------------------
	long double ratio = ((long double)succ / (long double)fail);
	long double diff = fabs(ratio - goldenNum);
	printf("\nMatches: %d\nFailed : %d\n\n", succ, fail);
	printf("Time  (sec)    : %f\n", seconds);
	//printf("Nearest match: %d\n", nearestMatch);
	printf("Achieved Ratio : %1.12Lf \nProbability    : %1.12Lf\nDifference     : %1.12Lf\n", ratio, goldenNum, diff);
	printf("---------------------------------------------\n");
	// -------------------------------------------------------
	// -------------------------------------------------------
	cudaDeviceReset();
	curandDestroyGenerator(genGPU);
	cudaFree(d_GPU);
	return 0;
}

void query_device()
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
	{
		printf("No CUDA support device found");
	}

	int devNo = 0;
	cudaDeviceProp iProp;
	cudaGetDeviceProperties(&iProp, devNo);

	printf("Device %d: %s\n", devNo, iProp.name);
	printf("  Number of multiprocessors:                     %d\n", iProp.multiProcessorCount);
	printf("  Clock rate:                                    %d\n",	iProp.clockRate);
	printf("  Compute capability:                            %d.%d\n", iProp.major, iProp.minor);
	printf("  Total amount of global memory:                 %4.2f KB\n", iProp.totalGlobalMem / 1024.0);
	printf("  Total amount of constant memory:               %4.2f KB\n", iProp.totalConstMem / 1024.0);
	printf("  Total amount of shared memory per block:       %4.2f KB\n", iProp.sharedMemPerBlock / 1024.0);
	printf("  Total amount of shared memory per MP:          %4.2f KB\n", iProp.sharedMemPerMultiprocessor / 1024.0);
	printf("  Total number of registers available per block: %d\n", iProp.regsPerBlock);
	printf("  Warp size:                                     %d\n", iProp.warpSize);
	printf("  Maximum number of threads per block:           %d\n", iProp.maxThreadsPerBlock);
	printf("  Maximum number of threads per multiprocessor:  %d\n", iProp.maxThreadsPerMultiProcessor);
	printf("  Maximum number of warps per multiprocessor:    %d\n", iProp.maxThreadsPerMultiProcessor / 32);
	printf("  Maximum Grid size:                             (%d,%d,%d)\n", iProp.maxGridSize[0], iProp.maxGridSize[1], iProp.maxGridSize[2]);
	printf("  Maximum block dimension:                       (%d,%d,%d)\n", iProp.maxThreadsDim[0], iProp.maxThreadsDim[1], iProp.maxThreadsDim[2]);
}
