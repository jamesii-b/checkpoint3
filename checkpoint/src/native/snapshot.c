#include "snapshot.h"
#include "allocator.h"
#include "config.h"
#include "types.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/file.h>

extern cudaError_t (*get_real_cudaMemcpy(void))(void*, const void*, size_t, enum cudaMemcpyKind);
extern cudaError_t (*get_real_cudaDeviceSynchronize(void))(void);
extern uint64_t interceptor_get_token_index(void);
extern uint64_t interceptor_get_rng_state(void);
extern void interceptor_set_token_index(uint64_t idx);
extern void interceptor_set_rng_state(uint64_t state);

static int current_rank = 0;
static int current_world_size = 1;

#define CONSOLIDATED_MAGIC 0x434B5054
#define CONSOLIDATED_VERSION 1

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t world_size;
    uint32_t num_ranks;
    uint64_t metadata_offset;
    uint64_t rank_table_offset;
} ConsolidatedHeader;

typedef struct {
    int32_t rank;
    uint64_t data_offset;
    uint64_t data_size;
} RankEntry;

void snapshot_set_rank(int rank) {
    current_rank = rank;
}

void snapshot_set_world_size(int world_size) {
    current_world_size = world_size;
}

int snapshot_get_rank(void) {
    return current_rank;
}

int snapshot_get_world_size(void) {
    return current_world_size;
}

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

static int lock_file(const char* path) {
    char lock_path[512];
    snprintf(lock_path, sizeof(lock_path), "%s.lock", path);
    int fd = open(lock_path, O_CREAT | O_RDWR, 0666);
    if (fd >= 0) {
        flock(fd, LOCK_EX);
    }
    return fd;
}

static void unlock_file(int fd) {
    if (fd >= 0) {
        flock(fd, LOCK_UN);
        close(fd);
    }
}

static int read_existing_checkpoint(const char* path, ConsolidatedHeader* header, RankEntry** entries, void*** rank_data) {
    FILE* f = fopen(path, "rb");
    if (!f) {
        return -1;
    }

    if (fread(header, sizeof(ConsolidatedHeader), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    if (header->magic != CONSOLIDATED_MAGIC) {
        fclose(f);
        return -1;
    }

    *entries = malloc(sizeof(RankEntry) * header->num_ranks);
    *rank_data = malloc(sizeof(void*) * header->num_ranks);

    fseek(f, header->rank_table_offset, SEEK_SET);
    fread(*entries, sizeof(RankEntry), header->num_ranks, f);

    for (uint32_t i = 0; i < header->num_ranks; i++) {
        (*rank_data)[i] = malloc((*entries)[i].data_size);
        fseek(f, (*entries)[i].data_offset, SEEK_SET);
        fread((*rank_data)[i], 1, (*entries)[i].data_size, f);
    }

    fclose(f);
    return 0;
}

int snapshot_save_distributed(const char* path, int rank, int world_size) {
    fprintf(stderr, "[SNAPSHOT] Creating distributed snapshot rank=%d/%d at token=%lu\n", 
            rank, world_size, interceptor_get_token_index());

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
            continue;
        }

        cudaError_t err = real_memcpy(host_buf, info->device_ptr, info->size, cudaMemcpyDeviceToHost);
        if (err == cudaSuccess) {
            buffers[buf_idx].device_ptr = info->device_ptr;
            buffers[buf_idx].size = info->size;
            buffers[buf_idx].host_data = host_buf;
            buf_idx++;
        } else {
            free(host_buf);
        }
    }

    allocator_unlock();

    size_t rank_data_size = sizeof(uint32_t) + sizeof(uint64_t) * 3;
    for (uint32_t i = 0; i < buf_idx; i++) {
        rank_data_size += sizeof(size_t) + buffers[i].size;
    }

    void* rank_data = malloc(rank_data_size);
    char* ptr = rank_data;

    uint64_t token_idx = interceptor_get_token_index();
    uint64_t rng = interceptor_get_rng_state();
    uint64_t epoch = 0;

    memcpy(ptr, &buf_idx, sizeof(uint32_t)); ptr += sizeof(uint32_t);
    memcpy(ptr, &epoch, sizeof(uint64_t)); ptr += sizeof(uint64_t);
    memcpy(ptr, &token_idx, sizeof(uint64_t)); ptr += sizeof(uint64_t);
    memcpy(ptr, &rng, sizeof(uint64_t)); ptr += sizeof(uint64_t);

    for (uint32_t i = 0; i < buf_idx; i++) {
        memcpy(ptr, &buffers[i].size, sizeof(size_t)); ptr += sizeof(size_t);
        memcpy(ptr, buffers[i].host_data, buffers[i].size); ptr += buffers[i].size;
        free(buffers[i].host_data);
    }
    free(buffers);

    int lock_fd = lock_file(path);

    ConsolidatedHeader header;
    RankEntry* existing_entries = NULL;
    void** existing_data = NULL;
    int has_existing = read_existing_checkpoint(path, &header, &existing_entries, &existing_data);

    FILE* f = fopen(path, "wb");
    if (!f) {
        unlock_file(lock_fd);
        free(rank_data);
        return -1;
    }

    uint32_t num_ranks = 1;
    if (has_existing == 0) {
        int found = 0;
        for (uint32_t i = 0; i < header.num_ranks; i++) {
            if (existing_entries[i].rank == rank) {
                found = 1;
                break;
            }
        }
        num_ranks = found ? header.num_ranks : header.num_ranks + 1;
    }

    ConsolidatedHeader new_header;
    new_header.magic = CONSOLIDATED_MAGIC;
    new_header.version = CONSOLIDATED_VERSION;
    new_header.world_size = world_size;
    new_header.num_ranks = num_ranks;
    new_header.metadata_offset = sizeof(ConsolidatedHeader);
    new_header.rank_table_offset = sizeof(ConsolidatedHeader);

    fwrite(&new_header, sizeof(ConsolidatedHeader), 1, f);

    RankEntry* new_entries = malloc(sizeof(RankEntry) * num_ranks);
    uint64_t data_offset = sizeof(ConsolidatedHeader) + sizeof(RankEntry) * num_ranks;

    int entry_idx = 0;
    if (has_existing == 0) {
        for (uint32_t i = 0; i < header.num_ranks; i++) {
            if (existing_entries[i].rank != rank) {
                new_entries[entry_idx].rank = existing_entries[i].rank;
                new_entries[entry_idx].data_offset = data_offset;
                new_entries[entry_idx].data_size = existing_entries[i].data_size;
                data_offset += existing_entries[i].data_size;
                entry_idx++;
            }
        }
    }

    new_entries[entry_idx].rank = rank;
    new_entries[entry_idx].data_offset = data_offset;
    new_entries[entry_idx].data_size = rank_data_size;
    entry_idx++;

    fwrite(new_entries, sizeof(RankEntry), num_ranks, f);

    if (has_existing == 0) {
        int write_idx = 0;
        for (uint32_t i = 0; i < header.num_ranks; i++) {
            if (existing_entries[i].rank != rank) {
                fwrite(existing_data[i], 1, existing_entries[i].data_size, f);
                write_idx++;
            }
        }
    }

    fwrite(rank_data, 1, rank_data_size, f);

    fclose(f);

    if (has_existing == 0) {
        for (uint32_t i = 0; i < header.num_ranks; i++) {
            free(existing_data[i]);
        }
        free(existing_data);
        free(existing_entries);
    }

    free(new_entries);
    free(rank_data);
    unlock_file(lock_fd);

    fprintf(stderr, "[SNAPSHOT] Distributed snapshot saved for rank %d\n", rank);
    return 0;
}

int snapshot_restore_distributed(const char* path, int rank, int world_size) {
    fprintf(stderr, "[SNAPSHOT] Restoring distributed snapshot rank=%d/%d\n", rank, world_size);

    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[SNAPSHOT] Failed to open file: %s\n", path);
        return -1;
    }

    ConsolidatedHeader header;
    if (fread(&header, sizeof(ConsolidatedHeader), 1, f) != 1) {
        fclose(f);
        return -1;
    }

    if (header.magic != CONSOLIDATED_MAGIC) {
        fprintf(stderr, "[SNAPSHOT] Invalid checkpoint file\n");
        fclose(f);
        return -1;
    }

    RankEntry* entries = malloc(sizeof(RankEntry) * header.num_ranks);
    fseek(f, header.rank_table_offset, SEEK_SET);
    fread(entries, sizeof(RankEntry), header.num_ranks, f);

    RankEntry* my_entry = NULL;
    for (uint32_t i = 0; i < header.num_ranks; i++) {
        if (entries[i].rank == rank) {
            my_entry = &entries[i];
            break;
        }
    }

    if (!my_entry) {
        fprintf(stderr, "[SNAPSHOT] No data found for rank %d\n", rank);
        free(entries);
        fclose(f);
        return -1;
    }

    fseek(f, my_entry->data_offset, SEEK_SET);

    uint32_t buffer_count;
    uint64_t saved_epoch, saved_token, saved_rng;

    fread(&buffer_count, sizeof(uint32_t), 1, f);
    fread(&saved_epoch, sizeof(uint64_t), 1, f);
    fread(&saved_token, sizeof(uint64_t), 1, f);
    fread(&saved_rng, sizeof(uint64_t), 1, f);

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
            free(entries);
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
                total_restored += size;
            }
        }

        free(host_data);
    }

    free(entries);
    fclose(f);

    fprintf(stderr, "[SNAPSHOT] Distributed restore complete for rank %d: %.1f MB\n", 
            rank, total_restored / (1024.0 * 1024.0));

    return 0;
}
