#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <curand_kernel.h>

__global__ void setup_kernel(curandState *state) {
    int id = threadIdx.x + blockIdx.x * 64;
    /* Each thread gets same seed, a different sequence number, no offset */
    curand_init(1234, id, 0, &state[id]);
}

__global__ void pod_racing(curandState *state, unsigned long int n, unsigned int *loss, unsigned int *win) {
    int id = threadIdx.x + blockIdx.x * 64;
    printf("%d, %d\n", threadIdx.x, blockIdx.x);
    unsigned int x;
    const unsigned int flips[] = { 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1 };

    /* Copy state to local memory for efficiency */
    curandState localState = state[id];
    
    /* Generate pseudo-random unsigned ints */
    for(int i = 0; i < n; i++) {
        x = curand(&localState);
        unsigned int subCount = 0;
        unsigned int flip = x % 2;

        if (flip != flips[subCount]) { // 0 based
            subCount = 0;
            loss[id] = 1;
        }
        else {
            subCount++;
            if (subCount == 16) { // 1 based
                subCount = 0;
                win[id] = 1;
            }
        }
    }

    /* Copy state back to global memory */
    state[id] = localState;
}

int main(int argc, char *argv[]) {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0)
        printf("No CUDA support device found");

    int devNo = 0;
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, devNo);

    // max grid size 
    dim3 grid = {p.maxGridSize[0]};
    dim3 block = {p.maxThreadsDim[0]};
    /****************************************************************************/
    int totalWin, totalLoss;
    curandState *devStates;
    unsigned int *d_loss, *d_win, *h_win, *h_loss;
    unsigned long int trials = 10000;

   /* https://www.sciencedirect.com/topics/computer-science/execution-configuration */
    
    /* Allocate space for results on host */
    h_win = (unsigned int *)calloc(grid * block, sizeof(int));
    h_loss = (unsigned int *)calloc(grid * block, sizeof(int));

    /* Allocate space for results on device */
    cudaMalloc((void **)&d_loss, grid * block * sizeof(unsigned int));
    cudaMalloc((void **)&d_win, grid * block * sizeof(unsigned int));
    
    // (set seed)
    cudaMalloc((void **)&devStates, 64 * 64 * sizeof(curandState));
    setup_kernel<<<grid, block>>>(devStates);


    totalWin = 0, totalLoss = 0;
    /* Set results to 0 */
    cudaMemset(d_loss, 0, grid * block * sizeof(unsigned int));
    cudaMemset(d_win, 0, grid * block * sizeof(unsigned int));

    // do the thing
    pod_racing<<<64, 64>>>(devStates, trials, d_loss, d_win);
    
    /* Copy device memory to host */
    cudaMemcpy(h_loss, d_loss, grid * block * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_win, d_win, grid * block * sizeof(unsigned int), cudaMemcpyDeviceToHost);

    /* Show result */
    
    for(int i = 0; i < grid * block; i++) {
        totalWin += h_win[i];
        totalLoss += h_loss[i];
    }

    printf("Total Win: %d\n", totalWin);
    printf("Total Loss: %d\n", totalLoss);

    /* Cleanup */
    cudaFree(devStates);
    cudaFree(h_win);
    cudaFree(h_loss);
    cudaFree(d_win);
    cudaFree(d_loss);
    return EXIT_SUCCESS;
}


