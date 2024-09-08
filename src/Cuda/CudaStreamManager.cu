#include "../../include/Cuda/CudaStreamManager.cuh"
#include <stdexcept>

CudaStreamManager& CudaStreamManager::get_instance() {
    static CudaStreamManager instance;
    return instance;
}

void CudaStreamManager::initialize_streams(int num_streams) {
    streams.resize(num_streams);
    for (int i = 0; i < num_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }
}

void CudaStreamManager::destroy_streams() {
    for (int i = 0; i < streams.size(); ++i) {
        cudaStreamDestroy(streams[i]);
    }
    streams.clear();
}

cudaStream_t CudaStreamManager::get_stream(int index) {
    if (index < 0 || index >= streams.size()) {
        throw std::out_of_range("Invalid stream index");
    }
    return streams[index];
}