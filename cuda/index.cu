#include <cuda.h>
#include <stdio.h>



__global__ void hello(){
    printf("Hello, CUDA!\n, threads=%d\n",threadIdx.x);

}

int main(){
    // hello<<<num_blocks, threads_per_block>>>();
// int tid = blockIdx.x * blockDim.x + threadIdx.x;

    return -1;
}