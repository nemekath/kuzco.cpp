#include <hip/hip_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    hipGetDeviceCount(&deviceCount);
    
    printf("=== Current P2P Status (before any EnablePeerAccess) ===\n");
    
    // Check if P2P is implicitly enabled by testing transfer speed
    // First: try a small transfer with hipMemcpyPeerAsync WITHOUT enabling P2P
    void *d0, *d1;
    size_t size = 64 * 1024 * 1024; // 64MB
    
    hipSetDevice(0);
    hipMalloc(&d0, size);
    hipSetDevice(1);
    hipMalloc(&d1, size);
    
    // Warmup
    hipMemcpyPeer(d1, 1, d0, 0, size);
    hipDeviceSynchronize();
    
    // Measure WITHOUT explicit EnablePeerAccess
    hipEvent_t start, stop;
    hipSetDevice(0);
    hipEventCreate(&start);
    hipEventCreate(&stop);
    
    hipEventRecord(start);
    for (int i = 0; i < 20; i++) {
        hipMemcpyPeer(d1, 1, d0, 0, size);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    float ms;
    hipEventElapsedTime(&ms, start, stop);
    double bw = ((double)size * 20 / (1024.0*1024.0*1024.0)) / (ms / 1000.0);
    printf("WITHOUT EnablePeerAccess: %.2f GB/s (%.1f ms)\n", bw, ms);
    
    // Now enable P2P
    hipSetDevice(0);
    hipError_t err0 = hipDeviceEnablePeerAccess(1, 0);
    hipSetDevice(1);
    hipError_t err1 = hipDeviceEnablePeerAccess(0, 0);
    printf("\nEnablePeerAccess 0->1: %s\n", err0 == hipSuccess ? "OK" : hipGetErrorString(err0));
    printf("EnablePeerAccess 1->0: %s\n", err1 == hipSuccess ? "OK" : hipGetErrorString(err1));
    
    // Warmup
    hipMemcpyPeer(d1, 1, d0, 0, size);
    hipDeviceSynchronize();
    
    // Measure WITH P2P
    hipSetDevice(0);
    hipEventRecord(start);
    for (int i = 0; i < 20; i++) {
        hipMemcpyPeer(d1, 1, d0, 0, size);
    }
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    
    hipEventElapsedTime(&ms, start, stop);
    bw = ((double)size * 20 / (1024.0*1024.0*1024.0)) / (ms / 1000.0);
    printf("WITH EnablePeerAccess:    %.2f GB/s (%.1f ms)\n", bw, ms);
    
    hipFree(d0);
    hipFree(d1);
    return 0;
}
