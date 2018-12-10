/*
 * 	A program that empirically estimates the probability of flipping 16 coins
 *      and getting the following sequence HHTHTHHHTHHHTHTH. The estimate should
 *      get better as more coins are flipped.
 *
 *	This implementation uses the CUDA random number generator API function
 * 	which uses the Marsenne Twister algorithm to generate random numbers.
 *      basic functionality: (Generate numbers -> checks numbers) -> loop
 *
 *	TODO: 1) Efficiency. The sets are checked on the host, perhaps it can be
 *       	 writen as a device kernel to optimize speed and resources...
 *	      2) Maybe add option to set run time in seconds?
 *
 *	Author: Joseph Osborne
 *	Date:	11/30/2018
 *
*/
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
	// Take input from command line
	unsigned long long int input = 0;
	if (argc < 2) {
		printf("USE ONE PARAMETER! How many times to run.\n");
		exit(1);
	}
	else {
		input = atol(argv[1]);
	}
	printf("---------------------------------------------\n");
	query_device(); // get device info and print
	printf("---------------------------------------------\n");
	printf("--- Flipping [%lld] sets of 16 coins ---\n", (input*1024));
	printf("---------------------------------------------\n\n");
	// -------------------------------------------------------
	// TIMER
	// counts how long program runs
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// START  -------------------------------------------------
	cudaEventRecord(start); // time start
	// -------------------------------------------------------
	// HHT HT HHHT HHHT HT H
  // H = 1
	// T = 0
	int succ = 0, fail = 0;
	// desired sequence
	const int flips[] = { 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1 };
	// -------------------------------------------------------
	// Mersenne Twister number generator
	// Each output element is a 32-bit unsigned int where all bits are random
	// generator initialization
	// set seed
	curandGenerator_t genGPU;
	curandCreateGenerator(&genGPU, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(genGPU, CURAND_ORDERING_PSEUDO_BEST);
	// "Hence the most efficient use of MTGP32 is to generate a multiple of 16384 samples."
	// -------------------------------------------------------
	// 1024 x 64 size array
	const int x = 16, y = 1024, n = x * y; // 16384
	unsigned int GPU[n], *d_GPU;
	// allocate space on gpu
	cudaMalloc(&d_GPU, n * sizeof(unsigned int));
	// -------------------------------------------------------
	printf("Now THIS is pod racing!\n");
	int offset;
	for (int z = 1; z <= input; z++) {
		// generate numbers
		// use generator genGPU at target d_GPU
		curandGenerate(genGPU, d_GPU, n);
		// cpy memory back to host
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
	// clean up
	curandDestroyGenerator(genGPU);
	cudaFree(d_GPU);
	return 0;
}

void query_device()
{
	int deviceCount = 0;
	cudaGetDeviceCount(&deviceCount);

	if (deviceCount == 0)
		printf("No CUDA support device found");

	int devNo = 0;
	cudaDeviceProp d_prop;
	cudaGetDeviceProperties(&d_prop, devNo);

	printf("Device %d: %s\n", devNo, d_prop.name);
	printf("  Number of multiprocessors:                     %d\n", d_prop.multiProcessorCount);
	printf("  Clock rate:                                    %d\n",	d_prop.clockRate);
	printf("  Compute capability:                            %d.%d\n", d_prop.major, d_prop.minor);
	printf("  Total amount of global memory:                 %4.2f KB\n", d_prop.totalGlobalMem / 1024.0);
}
