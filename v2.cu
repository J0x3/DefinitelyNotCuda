#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <assert.h>
#define randFact 10

long double goldenNum = powl((long double)(0.5), (long double)(16)); // (1/2)^16

__global__ void set_seed(curandState *state) {

	int idx = threadIdx.x + blockDim.x*blockIdx.x;
	curand_init(1234, idx, 0, &state[idx]); // CURAND_ORDERING_PSEUDO_BEST
}

__global__ void pod_racing(curandState *my_curandstate, const unsigned long long int n, unsigned long long int *succ, unsigned long long int *fail, unsigned long long int test) {

	int idx = threadIdx.x + blockDim.x*blockIdx.x;

	const int flips[] = { 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1 };

	int subCount = 0;
	int count = 0;
	while (count < n) {
		float randFloat = (curand_uniform(my_curandstate + idx) * randFact);
		int flip = (int)truncf(randFloat);
		flip = flip % 2;

		//printf("%d ", flip);
		test += 1;
		if (flip != flips[subCount]) { // 0 based
			subCount = 0;
			(test)++;
		}
		else {
			subCount++;
			if (subCount == 16) { // 1 based
				subCount = 0;
				succ[0]++;
			}
		}
		count++;
	}
	//printf("\n\n d_succ %d d_fail %d\n", succ, fail);
}

int main(int argc, char **argv) {

	unsigned long long int input = 0;
	if (argc < 2) {
		printf("USE ONE PARAMETER! How many times to run.\n");
		exit(1);
	}
	else {
		input = atol(argv[1]);
	}
	/****************************************************************************/
	// TIMER init
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// TIMER START  -------------------------------------------------
	cudaEventRecord(start);
	/****************************************************************************/
	curandState *d_state;
	cudaMalloc(&d_state, sizeof(curandState));

	size_t sz = sizeof(unsigned long long int);
	unsigned long long int *d_succ, *h_succ, *d_fail, *h_fail;
	unsigned long long int test = 0;
	// on your marks
	h_succ = (unsigned long long int*)malloc(sz);
	h_fail = (unsigned long long int*)malloc(sz);
	// ready
	cudaMalloc(&d_fail, sz);
	cudaMalloc(&d_succ, sz);
	// get set
	//cudaMemset(d_succ, 0, (1) * sizeof(unsigned long long int));
	//cudaMemset(d_fail, 0, (1) * sizeof(unsigned long long int));
	// go
	set_seed << <1, 1 >> > (d_state);
	// jk
	*h_succ = 0;
	*h_fail = 0;
	// get set
	cudaMemcpy(d_succ, h_succ, sz, cudaMemcpyHostToDevice);
	cudaMemcpy(d_fail, h_fail, sz, cudaMemcpyHostToDevice);
	// go
	pod_racing << <1, 1 >> > (d_state, input, d_succ, d_fail, test);
	// win
	cudaMemcpy(h_succ, d_succ, sz, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_fail, d_fail, sz, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	// TIMER STOP  -------------------------------------------------
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float seconds = (milliseconds / 1000);
	/****************************************************************************/
	printf("%lld ", test);
	unsigned long long int win = *h_succ, loss = *h_fail;
	printf("Time (sec): %f\n", seconds);
	printf("\n\n d_succ %d d_fail %d\n", win, loss);
	//printf("\n\n d_succ %d d_fail %d\n", h_succ[0], h_fail[0]);

	long double ratio = ((long double)win / (long double)loss);;
	long double diff = fabs(ratio - goldenNum);
	printf("Achieved Ratio : %1.12Lf \nProbability    : %1.12Lf\nDifference     : %1.12Lf\n", ratio, goldenNum, diff);

	cudaDeviceReset();
	return 0;
}
