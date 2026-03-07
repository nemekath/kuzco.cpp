#include <hip/hip_runtime.h>
#include <stdio.h>
#include <chrono>

#define SIZE (256 * 1024 * 1024)  // 256 MB
#define ITERATIONS 20

int main() {
    int canAccess;
    hipDeviceCanAccessPeer(&canAccess, 0, 1);
    if (!canAccess) { printf("No P2P\n"); return 1; }

    // Test 1: WITHOUT P2P (via host staging)
    printf("=== GPU0 -> GPU1 Transfer (256 MB, %d iterations) ===\n\n", ITERATIONS);

    void *d0, *d1, *host;
    hipSetDevice(0); hipMalloc(&d0, SIZE);
    hipSetDevice(1); hipMalloc(&d1, SIZE);
    hipHostMalloc(&host, SIZE, hipHostMallocDefault);

    // Warmup
    hipMemcpy(host, d0, SIZE, hipMemcpyDeviceToHost);
    hipMemcpy(d1, host, SIZE, hipMemcpyHostToDevice);

    // Via host staging
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        hipMemcpy(host, d0, SIZE, hipMemcpyDeviceToHost);
        hipMemcpy(d1, host, SIZE, hipMemcpyHostToDevice);
    }
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    double ms_staged = std::chrono::duration<double, std::milli>(end - start).count();
    double bw_staged = ((double)SIZE * ITERATIONS / (1024.0*1024.0*1024.0)) / (ms_staged / 1000.0);
    printf("Host-Staged:  %7.1f ms total, %6.2f GB/s\n", ms_staged, bw_staged);

    // Test 2: WITH P2P enabled
    hipSetDevice(0); hipDeviceEnablePeerAccess(1, 0);
    hipSetDevice(1); hipDeviceEnablePeerAccess(0, 0);

    // Warmup
    hipMemcpyPeer(d1, 1, d0, 0, SIZE);
    hipDeviceSynchronize();

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        hipMemcpyPeer(d1, 1, d0, 0, SIZE);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double ms_p2p = std::chrono::duration<double, std::milli>(end - start).count();
    double bw_p2p = ((double)SIZE * ITERATIONS / (1024.0*1024.0*1024.0)) / (ms_p2p / 1000.0);
    printf("P2P Direct:   %7.1f ms total, %6.2f GB/s\n", ms_p2p, bw_p2p);

    printf("\nSpeedup: %.2fx\n", bw_p2p / bw_staged);

    // Bidirectional
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < ITERATIONS; i++) {
        hipMemcpyPeer(d1, 1, d0, 0, SIZE);
        hipMemcpyPeer(d0, 0, d1, 1, SIZE);
    }
    hipDeviceSynchronize();
    end = std::chrono::high_resolution_clock::now();
    double ms_bidi = std::chrono::duration<double, std::milli>(end - start).count();
    double bw_bidi = ((double)SIZE * 2 * ITERATIONS / (1024.0*1024.0*1024.0)) / (ms_bidi / 1000.0);
    printf("Bidirectional: %7.1f ms total, %6.2f GB/s\n", ms_bidi, bw_bidi);

    hipFree(d0); hipFree(d1); hipHostFree(host);
    return 0;
}
