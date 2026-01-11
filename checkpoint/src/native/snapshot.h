#ifndef CUDA_SNAPSHOT_SNAPSHOT_H
#define CUDA_SNAPSHOT_SNAPSHOT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int snapshot_save(const char* path);
int snapshot_restore(const char* path);

int snapshot_save_distributed(const char* path, int rank, int world_size);
int snapshot_restore_distributed(const char* path, int rank, int world_size);

void snapshot_set_rank(int rank);
void snapshot_set_world_size(int world_size);
int snapshot_get_rank(void);
int snapshot_get_world_size(void);

#ifdef __cplusplus
}
#endif

#endif 
