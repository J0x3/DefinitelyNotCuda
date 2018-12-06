#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

__global__ void setup_kernel(curandState *state) {
    int id = threadIdx.x + blockIdx.x * 64;
    /* Each thread gets same seed, a different sequence 
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void generate_kernel(curandState *state, int n, unsigned int *result) {
    int id = threadIdx.x + blockIdx.x * 64;
    int count = 0;
    unsigned int x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random unsigned ints */
    for(int i = 0; i < n; i++) {
        x = curand(&localState);
	printf("%d ", x);
        /* Check if low bit set */
        if(x & 1) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_uniform_kernel(curandState *state, int n, unsigned int *result) {
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int count = 0;
    float x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random uniforms */
    for(int i = 0; i < n; i++) {
        x = curand_uniform(&localState);
        /* Check if > .5 */
        if(x > .5) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}

__global__ void generate_normal_kernel(curandState *state, int n, unsigned int *result) {
    int id = threadIdx.x + blockIdx.x * 64;
    unsigned int count = 0;
    float2 x;
    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    /* Generate pseudo-random normals */
    for(int i = 0; i < n/2; i++) {
        x = curand_normal2(&localState);
        /* Check if within one standard deviaton */
        if((x.x > -1.0) && (x.x < 1.0)) {
            count++;
        }
        if((x.y > -1.0) && (x.y < 1.0)) {
            count++;
        }
    }
    /* Copy state back to global memory */
    state[id] = localState;
    /* Store results */
    result[id] += count;
}


int main(int argc, char *argv[]) {
    int i;
    unsigned int total;
    curandState *devStates;
    unsigned int *devResults, *hostResults;
    int sampleCount = 10000;
    
    /* Allocate space for results on host */
    hostResults = (unsigned int *)calloc(64 * 64, sizeof(int));

    /* Allocate space for results on device */
    cudaMalloc((void **)&devResults, 64 * 64 * sizeof(unsigned int));

    /* Set results to 0 */
    cudaMemset(devResults, 0, 64 * 64 * sizeof(unsigned int));

    cudaMalloc((void **)&devStates, 64 * 64 * sizeof(curandState));
    setup_kernel<<<64, 64>>>(devStates);
    generate_kernel<<<64, 64>>>(devStates, sampleCount, devResults);
    
    /* Copy device memory to host */
    cudaMemcpy(hostResults, devResults, 64 * 64 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
    }
    //printf("%10.13f\n", (float)total / (64.0f * 64.0f * sampleCount * 50.0f));
        
    /* Set results to 0 */
    cudaMemset(devResults, 0, 64 * 64 * sizeof(unsigned int));

    /* Generate and use uniform pseudo-random  */
    generate_uniform_kernel<<<64, 64>>>(devStates, sampleCount, devResults);

    /* Copy device memory to host */
    cudaMemcpy(hostResults, devResults, 64 * 64 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
    }
    //printf("%10.13f\n", (float)total / (64.0f * 64.0f * sampleCount * 50.0f));
    
    /* Set results to 0 */
    cudaMemset(devResults, 0, 64 * 64 * sizeof(unsigned int));

    /* Generate and use normal pseudo-random  */
    generate_normal_kernel<<<64, 64>>>(devStates, sampleCount, devResults);

    /* Copy device memory to host */
    cudaMemcpy(hostResults, devResults, 64 * 64 * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /* Show result */
    total = 0;
    for(i = 0; i < 64 * 64; i++) {
        total += hostResults[i];
    }
    //printf("%10.13f\n", (float)total / (64.0f * 64.0f * sampleCount * 50.0f));

    /* Cleanup */
    cudaFree(devStates);
    cudaFree(devResults);
    free(hostResults);
    return EXIT_SUCCESS;
}