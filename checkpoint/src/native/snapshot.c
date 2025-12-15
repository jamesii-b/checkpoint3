#include "snapshot.h"
#include "allocator.h"
#include "config.h"
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

extern cudaError_t (*get_real_cudaMemcpy(void))(void*, const void*, size_t, enum cudaMemcpyKind);
extern cudaError_t (*get_real_cudaDeviceSynchronize(void))(void);
extern uint64_t interceptor_get_token_index(void);
extern uint64_t interceptor_get_rng_state(void);
extern void interceptor_set_token_index(uint64_t idx);
extern void interceptor_set_rng_state(uint64_t state);

int snapshot_save(const char* path) {
    fprintf(stderr, "[SNAPSHOT] Creating snapshot at token=%lu\n", 
            interceptor_get_token_index());
    
    allocator_classify_all(0);  
    
    cudaError_t (*real_sync)(void) = get_real_cudaDeviceSynchronize();
    if (real_sync) {
        real_sync();
    }
    
    allocator_lock();
    
    uint32_t persistent_count = 0;
    for (int i = 0; i < MAX_ALLOCATIONS; i++) {
        AllocationInfo* info = allocator_get_info(i);
        if (info && info->active && info->classification == BUFFER_PERSISTENT) {
            persistent_count++;
        }
    }
    
    fprintf(stderr, "[SNAPSHOT] Found %u persistent buffers to save\n", persistent_count);
    
    if (persistent_count == 0) {
        allocator_unlock();
        fprintf(stderr, "[SNAPSHOT] Warning: No persistent buffers found\n");
    }
    
    SnapshotBuffer* buffers = malloc(sizeof(SnapshotBuffer) * persistent_count);
    if (!buffers) {
        allocator_unlock();
        return -1;
    }
    
    cudaError_t (*real_memcpy)(void*, const void*, size_t, enum cudaMemcpyKind) = get_real_cudaMemcpy();
    uint32_t buf_idx = 0;
    
    for (int i = 0; i < MAX_ALLOCATIONS; i++) {
        AllocationInfo* info = allocator_get_info(i);
        if (!info || !info->active || info->classification != BUFFER_PERSISTENT) {
            continue;
        }
        
        void* host_buf = malloc(info->size);
        if (!host_buf) {
            fprintf(stderr, "[SNAPSHOT] Failed to allocate host buffer of size %zu\n", info->size);
            continue;
        }
        
        cudaError_t err = real_memcpy(host_buf, info->device_ptr, 
                                      info->size, cudaMemcpyDeviceToHost);
        
        if (err == cudaSuccess) {
            buffers[buf_idx].device_ptr = info->device_ptr;
            buffers[buf_idx].size = info->size;
            buffers[buf_idx].host_data = host_buf;
            
            fprintf(stderr, "[SNAPSHOT] Copied buffer %u: ptr=%p, size=%.1f MB\n", 
                    buf_idx, info->device_ptr, info->size / (1024.0 * 1024.0));
            buf_idx++;
        } else {
            fprintf(stderr, "[SNAPSHOT] Failed to copy buffer: ptr=%p, error=%d\n",
                    info->device_ptr, err);
            free(host_buf);
        }
    }
    
    allocator_unlock();
    
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[SNAPSHOT] Failed to open file: %s\n", path);
        for (uint32_t i = 0; i < buf_idx; i++) {
            free(buffers[i].host_data);
        }
        free(buffers);
        return -1;
    }
    
    uint64_t token_idx = interceptor_get_token_index();
    uint64_t rng = interceptor_get_rng_state();
    uint64_t epoch = 0;  
    
    fwrite(&buf_idx, sizeof(uint32_t), 1, f);
    fwrite(&epoch, sizeof(uint64_t), 1, f);
    fwrite(&token_idx, sizeof(uint64_t), 1, f);
    fwrite(&rng, sizeof(uint64_t), 1, f);
    
    size_t total_bytes = 0;
    for (uint32_t i = 0; i < buf_idx; i++) {
        fwrite(&buffers[i].size, sizeof(size_t), 1, f);
        fwrite(buffers[i].host_data, 1, buffers[i].size, f);
        total_bytes += buffers[i].size;
        free(buffers[i].host_data);
    }
    
    fclose(f);
    free(buffers);
    
    fprintf(stderr, "[SNAPSHOT] Saved %u buffers (%.1f MB) to %s\n", 
            buf_idx, total_bytes / (1024.0 * 1024.0), path);
    
    return 0;
}

int snapshot_restore(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[SNAPSHOT] Failed to open file: %s\n", path);
        return -1;
    }
    
    uint32_t buffer_count;
    uint64_t saved_epoch, saved_token, saved_rng;
    
    fread(&buffer_count, sizeof(uint32_t), 1, f);
    fread(&saved_epoch, sizeof(uint64_t), 1, f);
    fread(&saved_token, sizeof(uint64_t), 1, f);
    fread(&saved_rng, sizeof(uint64_t), 1, f);
    
    fprintf(stderr, "[SNAPSHOT] Restoring %u buffers from token=%lu\n", 
            buffer_count, saved_token);
    
    interceptor_set_token_index(saved_token);
    interceptor_set_rng_state(saved_rng);
    
    int used_slots[MAX_ALLOCATIONS] = {0};
    cudaError_t (*real_memcpy)(void*, const void*, size_t, enum cudaMemcpyKind) = get_real_cudaMemcpy();
    
    size_t total_restored = 0;
    
    for (uint32_t i = 0; i < buffer_count; i++) {
        size_t size;
        fread(&size, sizeof(size_t), 1, f);
        
        void* host_data = malloc(size);
        if (!host_data) {
            fprintf(stderr, "[SNAPSHOT] Failed to allocate host buffer\n");
            fclose(f);
            return -1;
        }
        
        fread(host_data, 1, size, f);
        
        allocator_lock();
        void* target_ptr = NULL;
        
        for (int j = 0; j < MAX_ALLOCATIONS; j++) {
            AllocationInfo* info = allocator_get_info(j);
            if (info && info->active && info->size == size && !used_slots[j]) {
                target_ptr = info->device_ptr;
                used_slots[j] = 1;
                break;
            }
        }
        
        allocator_unlock();
        
        if (target_ptr) {
            cudaError_t err = real_memcpy(target_ptr, host_data, size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                fprintf(stderr, "[SNAPSHOT] Restored buffer %u: ptr=%p, size=%.1f MB\n",
                        i, target_ptr, size / (1024.0 * 1024.0));
                total_restored += size;
            } else {
                fprintf(stderr, "[SNAPSHOT] Failed to restore buffer: error=%d\n", err);
            }
        } else {
            fprintf(stderr, "[SNAPSHOT] No matching allocation for buffer of size %zu\n", size);
        }
        
        free(host_data);
    }
    
    fclose(f);
    
    fprintf(stderr, "[SNAPSHOT] Restore complete: %.1f MB restored\n", 
            total_restored / (1024.0 * 1024.0));
    
    return 0;
}
