#include "allocator.h"
#include "config.h"
#include "types.h"
#include <stdio.h>
#include <string.h>
#include <pthread.h>

static AllocationInfo allocations[MAX_ALLOCATIONS];
static pthread_mutex_t alloc_mutex = PTHREAD_MUTEX_INITIALIZER;

void allocator_init(void) {
    pthread_mutex_lock(&alloc_mutex);
    memset(allocations, 0, sizeof(allocations));
    pthread_mutex_unlock(&alloc_mutex);
}

int allocator_find_slot(void* ptr) {
    for (int i = 0; i < MAX_ALLOCATIONS; i++) {
        if (allocations[i].active && allocations[i].device_ptr == ptr) {
            return i;
        }
    }
    return -1;
}

int allocator_find_free_slot(void) {
    for (int i = 0; i < MAX_ALLOCATIONS; i++) {
        if (!allocations[i].active) {
            return i;
        }
    }
    return -1;
}

void allocator_mark_access(void* ptr) {
    int idx = allocator_find_slot(ptr);
    if (idx >= 0) {
        allocations[idx].access_count++;
    }
}

void allocator_mark_access_at_epoch(void* ptr, uint64_t epoch) {
    pthread_mutex_lock(&alloc_mutex);
    int idx = allocator_find_slot(ptr);
    if (idx >= 0) {
        allocations[idx].last_access_epoch = epoch;
        allocations[idx].access_count++;
    }
    pthread_mutex_unlock(&alloc_mutex);
}

int allocator_register(void* ptr, size_t size, uint64_t epoch) {
    pthread_mutex_lock(&alloc_mutex);
    
    int existing = allocator_find_slot(ptr);
    if (existing >= 0) {
        allocations[existing].last_access_epoch = epoch;
        allocations[existing].access_count++;
        pthread_mutex_unlock(&alloc_mutex);
        return existing;
    }
    
    int slot = allocator_find_free_slot();
    if (slot >= 0) {
        allocations[slot].device_ptr = ptr;
        allocations[slot].size = size;
        allocations[slot].alloc_epoch = epoch;
        allocations[slot].last_access_epoch = epoch;
        allocations[slot].access_count = 0;
        allocations[slot].classification = BUFFER_UNKNOWN;
        allocations[slot].active = 1;
        
        fprintf(stderr, "[ALLOCATOR] Registered ptr=%p size=%zu MB at epoch=%lu\n", 
                ptr, size / (1024*1024), epoch);
    }
    
    pthread_mutex_unlock(&alloc_mutex);
    return slot;
}

int allocator_unregister(void* ptr) {
    pthread_mutex_lock(&alloc_mutex);
    
    int idx = allocator_find_slot(ptr);
    if (idx >= 0) {
        fprintf(stderr, "[ALLOCATOR] Unregistered ptr=%p size=%zu\n",
                ptr, allocations[idx].size);
        allocations[idx].active = 0;
        pthread_mutex_unlock(&alloc_mutex);
        return 0;
    }
    
    pthread_mutex_unlock(&alloc_mutex);
    return -1;
}

AllocationInfo* allocator_get_info(int index) {
    if (index < 0 || index >= MAX_ALLOCATIONS) {
        return NULL;
    }
    return &allocations[index];
}

int allocator_get_active_count(void) {
    pthread_mutex_lock(&alloc_mutex);
    
    int count = 0;
    for (int i = 0; i < MAX_ALLOCATIONS; i++) {
        if (allocations[i].active) {
            count++;
        }
    }
    
    pthread_mutex_unlock(&alloc_mutex);
    return count;
}

void allocator_classify_all(uint64_t current_epoch) {
    pthread_mutex_lock(&alloc_mutex);
    
    for (int i = 0; i < MAX_ALLOCATIONS; i++) {
        if (!allocations[i].active) continue;
        
        uint64_t lifetime = current_epoch - allocations[i].alloc_epoch;
        
        if (lifetime >= PERSISTENT_THRESHOLD_EPOCHS && 
            allocations[i].access_count >= ACCESS_COUNT_THRESHOLD &&
            allocations[i].size >= MIN_PERSISTENT_SIZE) {
            allocations[i].classification = BUFFER_PERSISTENT;
        } else if (lifetime < PERSISTENT_THRESHOLD_EPOCHS) {
            allocations[i].classification = BUFFER_UNKNOWN;
        } else {
            allocations[i].classification = BUFFER_EPHEMERAL;
        }
    }
    
    pthread_mutex_unlock(&alloc_mutex);
}

void allocator_print_stats(uint64_t current_epoch) {
    pthread_mutex_lock(&alloc_mutex);
    
    int active = 0, persistent = 0, ephemeral = 0, unknown = 0;
    size_t total_size = 0;
    
    for (int i = 0; i < MAX_ALLOCATIONS; i++) {
        if (allocations[i].active) {
            active++;
            total_size += allocations[i].size;
            
            switch (allocations[i].classification) {
                case BUFFER_PERSISTENT: persistent++; break;
                case BUFFER_EPHEMERAL: ephemeral++; break;
                case BUFFER_UNKNOWN: unknown++; break;
            }
        }
    }
    
    fprintf(stderr, "[ALLOCATOR] Epoch %lu: %d active (%.1f MB), "
            "%d persistent, %d ephemeral, %d unknown\n",
            current_epoch, active, total_size / (1024.0 * 1024.0),
            persistent, ephemeral, unknown);
    
    pthread_mutex_unlock(&alloc_mutex);
}

void allocator_lock(void) {
    pthread_mutex_lock(&alloc_mutex);
}

void allocator_unlock(void) {
    pthread_mutex_unlock(&alloc_mutex);
}
