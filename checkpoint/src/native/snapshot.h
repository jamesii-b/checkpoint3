#ifndef CUDA_SNAPSHOT_SNAPSHOT_H
#define CUDA_SNAPSHOT_SNAPSHOT_H

#ifdef __cplusplus
extern "C" {
#endif

int snapshot_save(const char* path);

int snapshot_restore(const char* path);

#ifdef __cplusplus
}
#endif

#endif 
