#ifndef CUBLAS_HANDLE_MANAGER_CUH
#define CUBLAS_HANDLE_MANAGER_CUH

#include <cublas_v2.h>
#include <vector>
#include <stdexcept>

class CublasHandleManager {
public:
    static CublasHandleManager& get_instance();

    void initialize_handles(int num_handles);
    void destroy_handles();
    cublasHandle_t get_handle(int index);

private:
    CublasHandleManager() = default;
    ~CublasHandleManager() = default;

    // Copying and assignment are prohibited
    CublasHandleManager(const CublasHandleManager&) = delete;
    CublasHandleManager& operator=(const CublasHandleManager&) = delete;

    std::vector<cublasHandle_t> handles;
};

#endif // CUBLAS_HANDLE_MANAGER_CUH
