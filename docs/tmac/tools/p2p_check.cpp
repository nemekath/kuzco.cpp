#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    printf("GPU Count: %d\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        printf("GPU %d: %s\n", i, prop.name);
        printf("  gcnArchName: %s\n", prop.gcnArchName);
    }
    printf("\n=== P2P Access Matrix ===\n");
    printf("%-6s", "");
    for (int j = 0; j < deviceCount; j++) printf("GPU%-4d", j);
    printf("\n");

    for (int i = 0; i < deviceCount; i++) {
        printf("GPU%-3d", i);
        for (int j = 0; j < deviceCount; j++) {
            if (i == j) { printf("  -   "); continue; }
            int canAccess;
            hipError_t err = hipDeviceCanAccessPeer(&canAccess, i, j);
            if (err != hipSuccess) {
                printf(" ERR  ");
            } else {
                printf("  %s  ", canAccess ? "YES" : "NO ");
            }
        }
        printf("\n");
    }

    printf("\n=== Attempting P2P Enable ===\n");
    for (int i = 0; i < deviceCount; i++) {
        for (int j = 0; j < deviceCount; j++) {
            if (i == j) continue;
            hipSetDevice(i);
            hipError_t err = hipDeviceEnablePeerAccess(j, 0);
            printf("GPU %d -> GPU %d: %s\n", i, j, 
                   err == hipSuccess ? "ENABLED" : hipGetErrorString(err));
        }
    }

    return 0;
}
