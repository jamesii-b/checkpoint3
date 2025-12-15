#ifndef CUDA_SNAPSHOT_ALLOCATOR_H
#define CUDA_SNAPSHOT_ALLOCATOR_H

#include "types.h"
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void allocator_init(void);

int allocator_find_slot(void* ptr);

int allocator_find_free_slot(void);

void allocator_mark_access(void* ptr);

int allocator_register(void* ptr, size_t size, uint64_t epoch);

int allocator_unregister(void* ptr);

AllocationInfo* allocator_get_info(int index);

int allocator_get_active_count(void);

void allocator_classify_all(uint64_t current_epoch);

void allocator_print_stats(uint64_t current_epoch);

void allocator_lock(void);
void allocator_unlock(void);

#ifdef __cplusplus
}
#endif

#endif 
