#define _GNU_SOURCE
#include "interceptor.h"
#include "allocator.h"
#include "config.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <cuda_runtime.h>

typedef cudaError_t (*cudaMalloc_t)(void**, size_t);
typedef cudaError_t (*cudaFree_t)(void*);
typedef cudaError_t (*cudaMemcpy_t)(void*, const void*, size_t, enum cudaMemcpyKind);
typedef cudaError_t (*cudaDeviceSynchronize_t)(void);
typedef cudaError_t (*cudaLaunchKernel_t)(const void*, dim3, dim3, void**, size_t, cudaStream_t);

static cudaMalloc_t real_cudaMalloc = NULL;
static cudaFree_t real_cudaFree = NULL;
static cudaMemcpy_t real_cudaMemcpy = NULL;
static cudaDeviceSynchronize_t real_cudaDeviceSynchronize = NULL;
static cudaLaunchKernel_t real_cudaLaunchKernel = NULL;

static uint64_t current_epoch = 0;
static uint64_t token_index = 0;
static uint64_t rng_state = 0;
static int initialized = 0;
static int interception_active = 0;

__attribute__((constructor))
static void early_init(void) {
    fprintf(stderr, "[INTERCEPTOR] Early initialization\n");
    interception_active = 1;
}

static void load_real_functions(void) {
    if (initialized) return;
    
    void* handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle) {
        handle = dlopen("libcudart.so.11", RTLD_LAZY | RTLD_GLOBAL);
    }
    if (!handle) {
        handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_GLOBAL);
    }
    
    real_cudaMalloc = (cudaMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");
    if (!real_cudaMalloc && handle) {
        real_cudaMalloc = (cudaMalloc_t)dlsym(handle, "cudaMalloc");
    }
    
    real_cudaFree = (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
    if (!real_cudaFree && handle) {
        real_cudaFree = (cudaFree_t)dlsym(handle, "cudaFree");
    }
    
    real_cudaMemcpy = (cudaMemcpy_t)dlsym(RTLD_NEXT, "cudaMemcpy");
    if (!real_cudaMemcpy && handle) {
        real_cudaMemcpy = (cudaMemcpy_t)dlsym(handle, "cudaMemcpy");
    }
    
    real_cudaDeviceSynchronize = (cudaDeviceSynchronize_t)dlsym(RTLD_NEXT, "cudaDeviceSynchronize");
    if (!real_cudaDeviceSynchronize && handle) {
        real_cudaDeviceSynchronize = (cudaDeviceSynchronize_t)dlsym(handle, "cudaDeviceSynchronize");
    }
    
    real_cudaLaunchKernel = (cudaLaunchKernel_t)dlsym(RTLD_NEXT, "cudaLaunchKernel");
    if (!real_cudaLaunchKernel && handle) {
        real_cudaLaunchKernel = (cudaLaunchKernel_t)dlsym(handle, "cudaLaunchKernel");
    }
    
    if (!real_cudaMalloc || !real_cudaFree || !real_cudaMemcpy || !real_cudaDeviceSynchronize) {
        fprintf(stderr, "[INTERCEPTOR] Warning: Some CUDA functions not resolved\n");
    }
    
    allocator_init();
    initialized = 1;
    fprintf(stderr, "[INTERCEPTOR] Initialization complete\n");
}

void interceptor_init(void) {
    load_real_functions();
}

void interceptor_shutdown(void) {
    initialized = 0;
}

void interceptor_tick_epoch(void) {
    __sync_add_and_fetch(&current_epoch, 1);
}

void interceptor_set_token_index(uint64_t idx) {
    token_index = idx;
}

uint64_t interceptor_get_token_index(void) {
    return token_index;
}

void interceptor_set_rng_state(uint64_t state) {
    rng_state = state;
}

uint64_t interceptor_get_rng_state(void) {
    return rng_state;
}

void interceptor_discover_allocations(void) {
    size_t free_mem, total_mem;
    cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
    
    if (err != cudaSuccess) {
        fprintf(stderr, "[INTERCEPTOR] Failed to get GPU memory info\n");
        return;
    }
    
    size_t used_mem = total_mem - free_mem;
    
    fprintf(stderr, "[INTERCEPTOR] GPU Memory: %.1f MB used, %.1f MB free, %.1f MB total\n",
            used_mem / (1024.0 * 1024.0), 
            free_mem / (1024.0 * 1024.0), 
            total_mem / (1024.0 * 1024.0));
    
    fprintf(stderr, "[INTERCEPTOR] Tracked %d allocations\n", 
            allocator_get_active_count());
}

void interceptor_register_allocation(void* ptr, size_t size) {
    if (!ptr || size == 0) return;
    allocator_register(ptr, size, current_epoch);
}

void interceptor_classify_buffers(void) {
    allocator_classify_all(current_epoch);
}

void interceptor_print_stats(void) {
    allocator_print_stats(current_epoch);
}


cudaError_t cudaMalloc(void** devPtr, size_t size) {
    if (!initialized) load_real_functions();
    
    if (!real_cudaMalloc) {
        void* handle = dlopen("libcudart.so", RTLD_LAZY | RTLD_NOLOAD);
        if (!handle) handle = dlopen("libcudart.so.11", RTLD_LAZY | RTLD_NOLOAD);
        if (!handle) handle = dlopen("libcudart.so.12", RTLD_LAZY | RTLD_NOLOAD);
        if (handle) real_cudaMalloc = (cudaMalloc_t)dlsym(handle, "cudaMalloc");
    }
    
    if (!real_cudaMalloc) {
        fprintf(stderr, "[INTERCEPTOR] Cannot find real cudaMalloc\n");
        return cudaErrorInitializationError;
    }
    
    cudaError_t ret = real_cudaMalloc(devPtr, size);
    
    if (ret == cudaSuccess && devPtr && *devPtr) {
        allocator_register(*devPtr, size, current_epoch);
    }
    
    return ret;
}

cudaError_t cudaFree(void* devPtr) {
    if (!initialized) load_real_functions();
    
    if (devPtr) {
        allocator_unregister(devPtr);
    }
    
    return real_cudaFree(devPtr);
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind) {
    if (!initialized) load_real_functions();
    
    if (kind == cudaMemcpyDeviceToHost || kind == cudaMemcpyHostToDevice) {
        void* device_ptr = (kind == cudaMemcpyDeviceToHost) ? (void*)src : dst;
        allocator_mark_access(device_ptr);
    }
    
    return real_cudaMemcpy(dst, src, count, kind);
}

cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, 
                             void** args, size_t sharedMem, cudaStream_t stream) {
    if (!initialized) load_real_functions();
    
    if (real_cudaLaunchKernel) {
        return real_cudaLaunchKernel(func, gridDim, blockDim, args, sharedMem, stream);
    }
    
    return cudaSuccess;
}

cudaMemcpy_t get_real_cudaMemcpy(void) {
    return real_cudaMemcpy;
}

cudaDeviceSynchronize_t get_real_cudaDeviceSynchronize(void) {
    return real_cudaDeviceSynchronize;
}
