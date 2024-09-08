#include "../../include/Cuda/CublasHandleManager.cuh"

CublasHandleManager& CublasHandleManager::get_instance() {
    static CublasHandleManager instance;
    return instance;
}

void CublasHandleManager::initialize_handles(int num_handles) {
    handles.resize(num_handles);
    for (int i = 0; i < num_handles; ++i) {
        cublasCreate(&handles[i]);
    }
}

void CublasHandleManager::destroy_handles() {
    for (int i = 0; i < handles.size(); ++i) {
        cublasDestroy(handles[i]);
    }
    handles.clear();
}

cublasHandle_t CublasHandleManager::get_handle(int index) {
    if (index < 0 || index >= handles.size()) {
        throw std::out_of_range("Invalid handle index");
    }
    return handles[index];
}
