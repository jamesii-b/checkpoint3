#ifndef CUDA_SNAPSHOT_TYPES_H
#define CUDA_SNAPSHOT_TYPES_H

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    BUFFER_EPHEMERAL = 0,   
    BUFFER_PERSISTENT = 1,  
    BUFFER_UNKNOWN = 2      
} BufferClass;

typedef struct {
    void* device_ptr;              
    size_t size;                   
    uint64_t alloc_epoch;          
    uint64_t last_access_epoch;    
    uint32_t access_count;         
    BufferClass classification;    
    int active;                    
} AllocationInfo;

typedef struct {
    void* device_ptr;   
    size_t size;        
    void* host_data;    
} SnapshotBuffer;

typedef struct {
    SnapshotBuffer* buffers;  
    uint32_t buffer_count;    
    uint64_t snapshot_epoch;  
    uint64_t token_index;     
    uint64_t rng_state;       
} SnapshotData;

#ifdef __cplusplus
}
#endif

#endif 
