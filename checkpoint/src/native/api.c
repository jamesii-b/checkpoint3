#include "interceptor.h"
#include "snapshot.h"

int interceptor_snapshot(const char* path) {
    return snapshot_save(path);
}

int interceptor_restore(const char* path) {
    return snapshot_restore(path);
}
