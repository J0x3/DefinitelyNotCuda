/*
 * 	A program that empirically estimates the probability of flipping 16 coins
 *      and getting the following sequence HHTHTHHHTHHHTHTH. The estimate should
 *      get better as more coins are flipped.
 *
 *	This implementation uses the CUDA random number generator API function
 * 	which uses the Marsenne Twister algorithm to generate random numbers.
 *
 *	Program sets up necessary memory and streams. Sets memory to 0.
 *  Generates random numbers. Calls kernel code streams to check flips.
 *  Counts all statistics. Breaks it all down and frees memory.
 *
 *	TODO: 1) Efficiency. Concurrency
 *	      2) Maybe add option to set run time in seconds?
 *
 *	Author: Joseph Osborne
 *	Date:	12/29/2018
 *
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <math.h>
//#include <time.h>

//const int num_streams = 4;

/*
struct strctStrm {
	cudaStream_t streams[num_streams];
	// index of flip set
	int *iter[num_streams];
	// randz
	unsigned int *h_rand[num_streams];
	unsigned int *d_rand[num_streams];
	// host
	unsigned int *h_win[num_streams];
	unsigned int *h_loss[num_streams];
	// device
	unsigned int *d_win[num_streams];
	unsigned int *d_loss[num_streams];
	// result
	unsigned int *h_resWIN[num_streams];
	unsigned int *h_resLOS[num_streams];
	unsigned int *d_resWIN[num_streams];
	unsigned int *d_resLOS[num_streams];

	// global
	unsigned long long int win[num_streams];
	unsigned long long int loss[num_streams];
};
*/

long double goldenNum = powl((long double)(0.5), (long double)(16)); // (1/2)^16
void query_device();

// kernel code for summing arrays
// may not end up using
__global__ void sum_arrays_gpu(unsigned int * a, unsigned int * b, int size)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (index < size) {
		a[0] = a[0] + b[index];
		//printf("%u ", a[0]);
	}
}

__global__ void pod_racing(unsigned int *d_rand, unsigned int *win, unsigned int *loss, unsigned int size, int *iter) {
	int index = threadIdx.x + blockDim.x*blockIdx.x;
	const unsigned int flips[] = { 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1 };
	if (index < size) {
		//printf("%d ", iter[0]);
		if ((d_rand[index] % 2) != flips[iter[0]]) {
			iter[0] = 0;
			loss[index] = 1;
			//printf("loss ");
		}
		else {
			iter[0] = iter[0] + 1;
			if (iter[0] == 15) {
				win[index] = 1;
				iter[0] = 0;
				//printf("win ");
			}
		}
	}
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
	query_device();
	// TIMER
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// START  -------------------------------------------------
	cudaEventRecord(start);
	// -------------------------------------------------------
	// Mersenne Twister number generator
	// Each output element is a 32-bit unsigned int where all bits are random
	curandGenerator_t generator;
	curandCreateGenerator(&generator, CURAND_RNG_PSEUDO_MTGP32);
	curandSetPseudoRandomGeneratorSeed(generator, CURAND_ORDERING_PSEUDO_BEST);
	// "Hence the most efficient use of MTGP32 is to generate a multiple of 16384 samples."
	// -------------------------------------------------------
	// max x dimension == 2^31 − 1 == 2147483647
	// 2146304 = 16384 * 131
	// 2097152 =  16384 * 128
	// 128 * 128 == 16384
	// (16 * 8) * 128 == 16384
	//const int size = 1 << 25;
	//const int block_size = 128;
	//const int x = (16), noRands = x * 256; // 16384
	const long int size = 2097152 / 64;
	size_t NUM_BYTES = size * sizeof(unsigned int);
	unsigned long long int win1 = 0, loss1 = 0, win2 = 0, loss2 = 0, win3 = 0, loss3 = 0, win4 = 0, loss4 = 0;
	unsigned long long int win5 = 0, loss5 = 0, win6 = 0, loss6 = 0, win7 = 0, loss7 = 0, win8 = 0, loss8 = 0;
	// -------------------------------------------------------
	unsigned int h_rand1[size], *d_rand1;
	unsigned int h_rand2[size], *d_rand2;
	unsigned int h_rand3[size], *d_rand3;
	unsigned int h_rand4[size], *d_rand4;
	unsigned int h_rand5[size], *d_rand5;
	unsigned int h_rand6[size], *d_rand6;
	unsigned int h_rand7[size], *d_rand7;
	//unsigned int h_rand8[size], *d_rand8;
	// -------------------------------------------------------
	cudaMalloc(&d_rand1, size * sizeof(unsigned int));
	cudaMalloc(&d_rand2, size * sizeof(unsigned int));
	cudaMalloc(&d_rand3, size * sizeof(unsigned int));
	cudaMalloc(&d_rand4, size * sizeof(unsigned int));
	cudaMalloc(&d_rand5, size * sizeof(unsigned int));
	cudaMalloc(&d_rand6, size * sizeof(unsigned int));
	cudaMalloc(&d_rand7, size * sizeof(unsigned int));
	//cudaMalloc(&d_rand8, size * sizeof(unsigned int));
	// -------------------------------------------------------
	int *iter1, *iter2, *iter3, *iter4;
	int *iter5, *iter6, *iter7, *iter8;
	cudaMalloc(&iter1, sizeof(int));
	cudaMalloc(&iter2, sizeof(int));
	cudaMalloc(&iter3, sizeof(int));
	cudaMalloc(&iter4, sizeof(int));
	cudaMalloc(&iter5, sizeof(int));
	cudaMalloc(&iter6, sizeof(int));
	cudaMalloc(&iter7, sizeof(int));
	//cudaMalloc(&iter8, sizeof(int));
	// -------------------------------------------------------
	unsigned int *h_win1, *h_loss1, *h_win2, *h_loss2, *h_win3, *h_loss3, *h_win4, *h_loss4;
	unsigned int *h_win5, *h_loss5, *h_win6, *h_loss6, *h_win7, *h_loss7, *h_win8, *h_loss8;
	h_win1 = (unsigned int *)malloc(NUM_BYTES);
	h_loss1 = (unsigned int *)malloc(NUM_BYTES);
	h_win2 = (unsigned int *)malloc(NUM_BYTES);
	h_loss2 = (unsigned int *)malloc(NUM_BYTES);
	h_win3 = (unsigned int *)malloc(NUM_BYTES);
	h_loss3 = (unsigned int *)malloc(NUM_BYTES);
	h_win4 = (unsigned int *)malloc(NUM_BYTES);
	h_loss4 = (unsigned int *)malloc(NUM_BYTES);
	h_win5 = (unsigned int *)malloc(NUM_BYTES);
	h_loss5 = (unsigned int *)malloc(NUM_BYTES);
	h_win6 = (unsigned int *)malloc(NUM_BYTES);
	h_loss6 = (unsigned int *)malloc(NUM_BYTES);
	h_win7 = (unsigned int *)malloc(NUM_BYTES);
	h_loss7 = (unsigned int *)malloc(NUM_BYTES);
	//h_win8 = (unsigned int *)malloc(NUM_BYTES);
	//h_loss8 = (unsigned int *)malloc(NUM_BYTES);
	// -------------------------------------------------------
	unsigned int *d_win1, *d_loss1, *d_win2, *d_loss2, *d_win3, *d_loss3, *d_win4, *d_loss4;
	unsigned int *d_win5, *d_loss5, *d_win6, *d_loss6, *d_win7, *d_loss7, *d_win8, *d_loss8;
	cudaMalloc((unsigned int **)&d_win1, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_loss1, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_win2, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_loss2, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_win3, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_loss3, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_win4, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_loss4, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_win5, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_loss5, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_win6, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_loss6, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_win7, NUM_BYTES);
	cudaMalloc((unsigned int **)&d_loss7, NUM_BYTES);
	//cudaMalloc((unsigned int **)&d_win8, NUM_BYTES);
	//cudaMalloc((unsigned int **)&d_loss8, NUM_BYTES);

	// ------------------------------------------------------
	// create streams
	cudaStream_t strm1, strm2, strm3, strm4, strm5, strm6, strm7, strm8;
	cudaStreamCreate(&strm1);
	cudaStreamCreate(&strm2);
	cudaStreamCreate(&strm3);
	cudaStreamCreate(&strm4);
	cudaStreamCreate(&strm5);
	cudaStreamCreate(&strm6);
	cudaStreamCreate(&strm7);
	//cudaStreamCreate(&strm8);
	// -------------------------------------------------------

	/*

	strctStrm *masterStream;
	masterStream = (strctStrm*)malloc(sizeof(masterStream));

	for (int i = 0; i < num_streams; i++) {
		//stream
		cudaStreamCreate(&masterStream[i].streams[i]);
		// allocation
		cudaMalloc(&masterStream[i].iter[i], sizeof(int));

		// rand
		masterStream[i].h_rand[i] = (unsigned int *)(calloc(size, sizeof(unsigned int)));
		cudaMalloc(&masterStream[i].d_rand[i], size * sizeof(unsigned int));

		// host
		masterStream[i].h_win[i] = (unsigned int *)malloc(NUM_BYTES);
		masterStream[i].h_loss[i] = (unsigned int *)malloc(NUM_BYTES);
		// device
		cudaMalloc((unsigned int **)&masterStream[i].d_win[i], NUM_BYTES);
		cudaMalloc((unsigned int **)&masterStream[i].d_loss[i], NUM_BYTES);

		// result
		masterStream[i].h_resWIN[i] = (unsigned int *)malloc(NUM_BYTES);
		masterStream[i].h_resLOS[i] = (unsigned int *)malloc(NUM_BYTES);
		cudaMalloc((unsigned int **)&masterStream[i].d_resWIN[i], NUM_BYTES);
		cudaMalloc((unsigned int **)&masterStream[i].d_resLOS[i], NUM_BYTES);
	}

	*/
	// run cycle of code input times
	for (int j = 0; j < input; j++) {
		/*
		for (int k = 0; k < num_streams; k++) {
			curandGenerate(generator, masterStream[k].d_rand[k], size);
			cudaMemcpy(masterStream[k].h_rand[k], masterStream[k].d_rand, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);

			cudaMemset(masterStream[k].iter[k], 0, sizeof(int));

			memset(masterStream[k].h_win[k], 0, NUM_BYTES);
			memset(masterStream[k].h_loss[k], 0, NUM_BYTES);
			memset(masterStream[k].h_resWIN[k], 0, NUM_BYTES);
			memset(masterStream[k].h_resLOS[k], 0, NUM_BYTES);

			cudaMemset(masterStream[k].d_win[k], 0, NUM_BYTES);
			cudaMemset(masterStream[k].d_loss[k], 0, NUM_BYTES);
			cudaMemset(masterStream[k].d_resWIN[k], 0, NUM_BYTES);
			cudaMemset(masterStream[k].d_resLOS[k], 0, NUM_BYTES);


			// H -> D
			cudaMemcpy(masterStream[k].d_win[k], masterStream[k].h_win[k], NUM_BYTES, cudaMemcpyHostToDevice);
			cudaMemcpy(masterStream[k].d_loss[k], masterStream[k].h_loss[k], NUM_BYTES, cudaMemcpyHostToDevice);

			pod_racing << <size, 1, 0, masterStream[k].streams[k] >> > (masterStream[k].d_rand[k],
																		masterStream[k].d_win[k],
																		masterStream[k].d_loss[k],
																		size,
																		masterStream[k].iter[k]);

			cudaDeviceSynchronize();
			// H -> D
			//cudaMemcpy(masterStream[n].d_win[n], masterStream[n].h_win[n], NUM_BYTES, cudaMemcpyHostToDevice);
			//cudaMemcpy(masterStream[n].d_loss[n], masterStream[n].h_loss[n], NUM_BYTES, cudaMemcpyHostToDevice);

			//cudaMemcpy(masterStream[k].d_resWIN[k], masterStream[k].h_resWIN[k], NUM_BYTES, cudaMemcpyHostToDevice);
			//cudaMemcpy(masterStream[k].d_resLOS[k], masterStream[k].h_resLOS[k], NUM_BYTES, cudaMemcpyHostToDevice);


			//sum_arrays_gpu << <size, 1 >> > (masterStream[k].d_resWIN[k], masterStream[k].d_win[k], size);
			//sum_arrays_gpu << <size, 1 >> > (masterStream[k].d_resLOS[k], masterStream[k].d_loss[k], size);

			//cudaDeviceSynchronize();

			// H <- D
			//cudaMemcpy(masterStream[n].h_win[n], masterStream[n].d_win[n], NUM_BYTES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(masterStream[n].h_loss[n], masterStream[n].d_loss[n], NUM_BYTES, cudaMemcpyDeviceToHost);

			//cudaStreamSynchronize(masterStream[k].streams[k]);
			// H <- D
			cudaMemcpy(masterStream[k].h_win[k], masterStream[k].d_win[k], NUM_BYTES, cudaMemcpyDeviceToHost);
			cudaMemcpy(masterStream[k].h_loss[k], masterStream[k].d_loss[k], NUM_BYTES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(masterStream[k].h_resWIN[k], masterStream[k].d_resWIN[k], NUM_BYTES, cudaMemcpyDeviceToHost);
			//cudaMemcpy(masterStream[k].h_resLOS[k], masterStream[k].d_resLOS[k], NUM_BYTES, cudaMemcpyDeviceToHost);


			//printf("win: %u\n", masterStream[k].h_resWIN[k][0]);
			//printf("loss: %u\n", masterStream[k].h_resLOS[k][0]);
		}
		*/
		// generate numbers for each stream and copy data back to host
		curandGenerate(generator, d_rand1, size);
		//cudaMemcpy(h_rand1, d_rand1, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//curandGenerate(generator, d_rand2, size);
		//cudaMemcpy(h_rand2, d_rand2, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//curandGenerate(generator, d_rand3, size);
		//cudaMemcpy(h_rand3, d_rand3, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//curandGenerate(generator, d_rand4, size);
		//cudaMemcpy(h_rand4, d_rand4, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//curandGenerate(generator, d_rand5, size);
		//cudaMemcpy(h_rand5, d_rand5, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//curandGenerate(generator, d_rand6, size);
		//cudaMemcpy(h_rand6, d_rand6, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//curandGenerate(generator, d_rand7, size);
		//cudaMemcpy(h_rand7, d_rand7, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		//curandGenerate(generator, d_rand8, size);
		//cudaMemcpy(h_rand8, d_rand8, size * sizeof(unsigned int), cudaMemcpyDeviceToHost);
		// -------------------------------------------------------
		// iter is the iterator of flip set
		// if it reaches 16, a whole set has been flipped correctly
		// iter for each stream, init to 0
		cudaMemset(iter1, 0, sizeof(int));
		cudaMemset(iter2, 0, sizeof(int));
		cudaMemset(iter3, 0, sizeof(int));
		cudaMemset(iter4, 0, sizeof(int));
		cudaMemset(iter5, 0, sizeof(int));
		cudaMemset(iter6, 0, sizeof(int));
		cudaMemset(iter7, 0, sizeof(int));
		//cudaMemset(iter8, 0, sizeof(int));
		// -------------------------------------------------------
		// set host memory to 0
		memset(h_win1, 0, NUM_BYTES);
		memset(h_loss1, 0, NUM_BYTES);
		memset(h_win2, 0, NUM_BYTES);
		memset(h_loss2, 0, NUM_BYTES);
		memset(h_win3, 0, NUM_BYTES);
		memset(h_loss3, 0, NUM_BYTES);
		memset(h_win4, 0, NUM_BYTES);
		memset(h_loss4, 0, NUM_BYTES);
		memset(h_win5, 0, NUM_BYTES);
		memset(h_loss5, 0, NUM_BYTES);
		memset(h_win6, 0, NUM_BYTES);
		memset(h_loss6, 0, NUM_BYTES);
		memset(h_win7, 0, NUM_BYTES);
		memset(h_loss7, 0, NUM_BYTES);
		//memset(h_win8, 0, NUM_BYTES);
		//memset(h_loss8, 0, NUM_BYTES);
		// set device memory to 0
		cudaMemset(d_win1, 0, NUM_BYTES);
		cudaMemset(d_loss1, 0, NUM_BYTES);
		cudaMemset(d_win2, 0, NUM_BYTES);
		cudaMemset(d_loss2, 0, NUM_BYTES);
		cudaMemset(d_win3, 0, NUM_BYTES);
		cudaMemset(d_loss3, 0, NUM_BYTES);
		cudaMemset(d_win4, 0, NUM_BYTES);
		cudaMemset(d_loss4, 0, NUM_BYTES);
		cudaMemset(d_win5, 0, NUM_BYTES);
		cudaMemset(d_loss5, 0, NUM_BYTES);
		cudaMemset(d_win6, 0, NUM_BYTES);
		cudaMemset(d_loss6, 0, NUM_BYTES);
		cudaMemset(d_win7, 0, NUM_BYTES);
		cudaMemset(d_loss7, 0, NUM_BYTES);
		//cudaMemset(d_win8, 0, NUM_BYTES);
		//cudaMemset(d_loss8, 0, NUM_BYTES);

		// H -> D
		// copy memory from host to device
		cudaMemcpy(d_win1, h_win1, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_loss1, h_loss1, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_win2, h_win2, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_loss2, h_loss2, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_win3, h_win3, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_loss3, h_loss3, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_win4, h_win4, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_loss4, h_loss4, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_win5, h_win5, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_loss5, h_loss5, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_win6, h_win6, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_loss6, h_loss6, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_win7, h_win7, NUM_BYTES, cudaMemcpyHostToDevice);
		cudaMemcpy(d_loss7, h_loss7, NUM_BYTES, cudaMemcpyHostToDevice);
		//cudaMemcpy(d_win8, h_win8, NUM_BYTES, cudaMemcpyHostToDevice);
		//cudaMemcpy(d_loss8, h_loss8, NUM_BYTES, cudaMemcpyHostToDevice);
		//cudaMemcpy(d_rand1, h_rand1, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_rand2, h_rand2, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_rand3, h_rand3, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_rand4, h_rand4, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_rand5, h_rand5, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_rand6, h_rand6, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_rand7, h_rand7, size * sizeof(unsigned int), cudaMemcpyHostToDevice);
		//cudaMemcpy(d_rand8, h_rand8, size * sizeof(unsigned int), cudaMemcpyHostToDevice);


		// run kernel code
		// sends parameters to device
		// basically function call, but to the GPU
		// this version uses streams and running code on multiple instances (streams)
		pod_racing << <size, 1, 0, strm1 >> > (d_rand1, d_win1, d_loss1, size, iter1);
		pod_racing << <size, 1, 0, strm2 >> > (d_rand1, d_win2, d_loss2, size, iter2);
		pod_racing << <size, 1, 0, strm3 >> > (d_rand1, d_win3, d_loss3, size, iter3);
		pod_racing << <size, 1, 0, strm4 >> > (d_rand1, d_win4, d_loss4, size, iter4);
		pod_racing << <size, 1, 0, strm5 >> > (d_rand1, d_win5, d_loss5, size, iter5);
		pod_racing << <size, 1, 0, strm6 >> > (d_rand1, d_win6, d_loss6, size, iter6);
		pod_racing << <size, 1, 0, strm7 >> > (d_rand1, d_win7, d_loss7, size, iter7);
		//pod_racing << <size, 1, 0, strm8 >> > (d_rand8, d_win8, d_loss8, size, iter8);
		cudaDeviceSynchronize();

		// H <- D
		// copies the memory from the device back to host
		cudaMemcpy(h_win1, d_win1, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_loss1, d_loss1, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_win2, d_win2, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_loss2, d_loss2, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_win3, d_win3, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_loss3, d_loss3, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_win4, d_win4, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_loss4, d_loss4, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_win5, d_win5, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_loss5, d_loss5, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_win6, d_win6, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_loss6, d_loss6, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_win7, d_win7, NUM_BYTES, cudaMemcpyDeviceToHost);
		cudaMemcpy(h_loss7, d_loss7, NUM_BYTES, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_win8, d_win8, NUM_BYTES, cudaMemcpyDeviceToHost);
		//cudaMemcpy(h_loss8, d_loss8, NUM_BYTES, cudaMemcpyDeviceToHost);

		// add up the values generated by the kernels
		// 0 or 1 stored in each index
		// runs through them and add together
		for (int x = 0; x < size; x++) {
			win1 += h_win1[x];
			loss1 += h_loss1[x];
			win2 += h_win2[x];
			loss2 += h_loss2[x];
			win3 += h_win3[x];
			loss3 += h_loss3[x];
			win4 += h_win4[x];
			loss4 += h_loss4[x];
			win5 += h_win5[x];
			loss5 += h_loss5[x];
			win6 += h_win6[x];
			loss6 += h_loss6[x];
			win7 += h_win7[x];
			loss7 += h_loss7[x];
			//win8 += h_win8[x];
			//loss8 += h_loss8[x];
		}
		/*
		for (int x = 0; x < size; x++) {
			for (int l = 0; l < num_streams; l++) {
				masterStream[l].win[l] += masterStream[l].h_win[l];
				masterStream[l].loss[l] += masterStream[l].h_loss[l];
			}
		}

		for (int x = 0; x < size; x++) {
			for (int n = 0; n < num_streams; n++) {
				// H -> D
				cudaMemcpy(masterStream[n].d_resWIN[n], masterStream[n].h_resWIN[n], NUM_BYTES, cudaMemcpyHostToDevice);
				cudaMemcpy(masterStream[n].d_resLOS[n], masterStream[n].h_resLOS[n], NUM_BYTES, cudaMemcpyHostToDevice);

				sum_arrays_gpu << <size, 1 >> > (masterStream[n].d_resWIN[n], masterStream[n].d_win[n], size);
				sum_arrays_gpu << <size, 1 >> > (masterStream[n].d_resLOS[n], masterStream[n].d_loss[n], size);

				cudaDeviceSynchronize();
				// H <- D
				cudaMemcpy(masterStream[n].h_resWIN[n], masterStream[n].d_resWIN[n], NUM_BYTES, cudaMemcpyDeviceToHost);
				cudaMemcpy(masterStream[n].h_resLOS[n], masterStream[n].d_resLOS[n], NUM_BYTES, cudaMemcpyDeviceToHost);

				//printf("win: %u\n", masterStream[n].h_resWIN[n][0]);
				//printf("loss: %u\n", masterStream[n].h_resLOS[n][0]);
			}
		}
		for (int x = 0; x < size; x++) {
			for (int n = 0; n < num_streams; n++) {
				masterStream[n].win[n] =+ masterStream[n].h_win[n]
			win1 += h_win1[x];
			loss1 += h_loss1[x];
			win2 += h_win2[x];
			loss2 += h_loss2[x];
			win3 += h_win3[x];
			loss3 += h_loss3[x];
			win4 += h_win4[x];
			loss4 += h_loss4[x];
		}*/
	}
	/*
	unsigned long long int tW = 0, tL = 0;

	for (int t = 0; t < num_streams; t++) {
		tW += masterStream[t].win[t];
		tL += masterStream[t].loss[t];

	}
	printf("win: %llu\n", tW);
	printf("loss: %llu\n", tL);


	for (int m = 0; m < num_streams; m++) {
		cudaStreamDestroy(masterStream[m].streams[m]);
		//printf("%llu %llu\n", masterStream[m].win[m], masterStream[m].win[m]);
	}*/


	// deconstruct streams
	cudaStreamDestroy(strm1);
	cudaStreamDestroy(strm2);
	cudaStreamDestroy(strm3);
	cudaStreamDestroy(strm4);
	cudaStreamDestroy(strm5);
	cudaStreamDestroy(strm6);
	cudaStreamDestroy(strm7);
	//cudaStreamDestroy(strm8);

	// print out run statistics
	cudaDeviceSynchronize();
	// STOP  -------------------------------------------------
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	float seconds = (milliseconds / 1000);
	printf("---------------------------------------------\n");
	// -------------------------------------------------------
	printf("%llu %llu\n", win1, loss1);
	printf("%llu %llu\n", win2, loss2);
	printf("%llu %llu\n", win3, loss3);
	printf("%llu %llu\n", win4, loss4);
	printf("%llu %llu\n", win5, loss5);
	printf("%llu %llu\n", win6, loss6);
	printf("%llu %llu\n", win7, loss7);
	//printf("%llu %llu\n", win8, loss8);
	printf("Time  (sec)    : %f\n", seconds);
	printf("Total: %llu\n", (win1+win2+win3+win4+win5+win6+win7+win8)+(loss1+loss2+loss3+loss4+loss5+loss6+loss7+loss8));
	long double ratio = ((long double)(win1 + win2 + win3 + win4 + win5 + win6 + win7 + win8)
		/ (long double)(loss1 + loss2 + loss3 + loss4 + loss5 + loss6 + loss7 + loss8));
	printf("Achieved Ratio : %1.12Lf \nProbability    : %1.12Lf\n", ratio, goldenNum);

	//cudaDeviceReset();
	/*
	for (int n = 0; n < num_streams; n++) {
		cudaFree(&masterStream[n].iter[n]);

		free(&masterStream[n].h_rand[n]);
		cudaFree(&masterStream[n].d_rand[n]);

		free(&masterStream[n].h_win[n]);
		free(&masterStream[n].h_loss[n]);

		cudaFree(&masterStream[n].d_win[n]);
		cudaFree(&masterStream[n].d_loss[n]);
	}
	free(masterStream);*/

	/*******************************/
	/*Free all memory allocations */
	cudaDeviceReset();
	cudaFree(h_rand1);
	cudaFree(h_rand2);
	cudaFree(h_rand3);
	cudaFree(h_rand4);
	cudaFree(h_win1);
	cudaFree(h_loss1);
	cudaFree(h_win2);
	cudaFree(h_loss2);
	cudaFree(h_win3);
	cudaFree(h_loss3);
	cudaFree(h_win4);
	cudaFree(h_loss4);
	cudaFree(d_rand1);
	cudaFree(d_rand2);
	cudaFree(d_rand3);
	cudaFree(d_rand4);
	cudaFree(d_win1);
	cudaFree(d_loss1);
	cudaFree(d_win2);
	cudaFree(d_loss2);
	cudaFree(d_win3);
	cudaFree(d_loss3);
	cudaFree(d_win4);
	cudaFree(d_loss4);

	cudaFree(h_rand5);
	cudaFree(h_rand6);
	cudaFree(h_rand7);
	//cudaFree(h_rand8);
	cudaFree(h_win5);
	cudaFree(h_loss5);
	cudaFree(h_win6);
	cudaFree(h_loss6);
	cudaFree(h_win7);
	cudaFree(h_loss7);
	//cudaFree(h_win8);
	//cudaFree(h_loss8);
	cudaFree(d_rand5);
	cudaFree(d_rand6);
	cudaFree(d_rand7);
	//cudaFree(d_rand8);
	cudaFree(d_win5);
	cudaFree(d_loss5);
	cudaFree(d_win6);
	cudaFree(d_loss6);
	cudaFree(d_win7);
	cudaFree(d_loss7);
	//cudaFree(d_win8);
	//cudaFree(d_loss8);

	free(h_win1);
	free(h_loss1);
	free(h_win2);
	free(h_loss2);
	free(h_win3);
	free(h_loss3);
	free(h_win4);
	free(h_loss4);
	free(h_win5);
	free(h_loss5);
	free(h_win6);
	free(h_loss6);
	free(h_win7);
	free(h_loss7);
	//free(h_win8);
	//free(h_loss8);
	return EXIT_SUCCESS;
}

// get device info and print
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
