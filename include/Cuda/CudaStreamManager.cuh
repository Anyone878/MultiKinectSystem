#ifndef CUDA_STREAM_MANAGER_H
#define CUDA_STREAM_MANAGER_H

#include <cuda_runtime.h>
#include <vector>

class CudaStreamManager {
public:
    static CudaStreamManager& get_instance();

    void initialize_streams(int num_streams);
    void destroy_streams();

    cudaStream_t get_stream(int index);

private:
    CudaStreamManager() = default;
    ~CudaStreamManager() = default;

    CudaStreamManager(const CudaStreamManager&) = delete;
    CudaStreamManager& operator=(const CudaStreamManager&) = delete;

    std::vector<cudaStream_t> streams;
};

#endif // CUDA_STREAM_MANAGER_H
