#ifndef CUDA_SNAPSHOT_INTERCEPTOR_H
#define CUDA_SNAPSHOT_INTERCEPTOR_H

#include "types.h"
#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void interceptor_init(void);

void interceptor_shutdown(void);

void interceptor_tick_epoch(void);

void interceptor_set_token_index(uint64_t idx);

uint64_t interceptor_get_token_index(void);

void interceptor_set_rng_state(uint64_t state);

uint64_t interceptor_get_rng_state(void);

void interceptor_discover_allocations(void);

void interceptor_register_allocation(void* ptr, size_t size);

void interceptor_classify_buffers(void);

void interceptor_print_stats(void);

int interceptor_snapshot(const char* path);

int interceptor_restore(const char* path);

#ifdef __cplusplus
}
#endif

#endif 
