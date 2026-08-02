// The CUDA backend when the engine is built without CUDA.
//
// This is not filler: it is what lets engine/cuda.hpp be part of the public API
// without forcing anyone to install the toolkit, and what makes the dispatch in
// src/tensor.cpp an ordinary condition instead of an #ifdef per operation. Every
// operation returns false -- "I did not do it" -- and the caller carries on down
// the CPU path.

#ifndef ENGINE_CUDA

#include "engine/cuda.hpp"
#include "engine/detail/cuda_ops.hpp"

#include <stdexcept>

namespace engine {
namespace cuda {

bool available() {
    return false;
}
bool enabled() {
    return false;
}
void set_enabled(bool) {}

DeviceInfo device_info() {
    return DeviceInfo{};
}
void synchronize() {}

int compiled_version() {
    return 0;
}
int runtime_version() {
    return 0;
}
int driver_version() {
    return 0;
}
size_t kernels_launched() {
    return 0;
}
size_t kernels_failed() {
    return 0;
}
void reset_kernel_counters() {}

double peak_fp32_gflops() {
    return 0.0;
}
double peak_bandwidth_gbs() {
    return 0.0;
}

std::vector<KernelOccupancy> kernel_occupancy() {
    return {};
}

MatmulKernel matmul_kernel() {
    return MatmulKernel::Auto;
}
void set_matmul_kernel(MatmulKernel) {}
MatmulKernel resolve_matmul_kernel(size_t, size_t, size_t) {
    return MatmulKernel::Auto;
}
bool tensor_cores_available() {
    return false;
}

// This one does return something useful without CUDA: the benchmark and the
// tests print the variant's name even with no device.
const char* matmul_kernel_name(MatmulKernel kernel) {
    switch (kernel) {
        case MatmulKernel::Auto:
            return "auto";
        case MatmulKernel::Naive:
            return "naive";
        case MatmulKernel::Tiled:
            return "tiled";
        case MatmulKernel::RegisterTiled:
            return "register";
        case MatmulKernel::Vectorized:
            return "vectorized";
        case MatmulKernel::TensorCore:
            return "tensorcore";
        case MatmulKernel::TensorCoreFp16:
            return "tensorcore-fp16";
    }
    return "unknown";
}

size_t min_matmul_flops() {
    return 0;
}
size_t min_elementwise_elements() {
    return 0;
}
void set_thresholds(size_t, size_t) {}

TransferStats transfer_stats() {
    return TransferStats{};
}
void reset_transfer_stats() {}

namespace detail {

// Storage only calls these when somebody asks for the device side, and without
// ENGINE_CUDA that code is not even compiled. Reaching here would be a
// programming error, not a runtime condition.
void note_kernel_launched() {}
void note_kernel_failed() {}

float* device_alloc(size_t) {
    throw std::logic_error("The engine was built without CUDA: there is no device memory.");
}
void device_free(float*) {}
void copy_to_device(float*, const float*, size_t) {
    throw std::logic_error("The engine was built without CUDA: there is no device memory.");
}
void copy_to_host(float*, const float*, size_t) {
    throw std::logic_error("The engine was built without CUDA: there is no device memory.");
}
void copy_device_to_device(float*, const float*, size_t) {
    throw std::logic_error("The engine was built without CUDA: there is no device memory.");
}

}  // namespace detail

namespace ops {

bool binary(Binary, const Storage&, const Storage&, Storage&, size_t, size_t) {
    return false;
}
bool matmul(const Storage&, const Storage&, Storage&, size_t, size_t, size_t, size_t, bool, bool) {
    return false;
}
bool scalar(const Storage&, Storage&, float, float) {
    return false;
}
bool im2col(const Storage&, Storage&, const WindowShape&) {
    return false;
}
bool col2im(const Storage&, Storage&, const WindowShape&) {
    return false;
}
bool maxpool(const Storage&, Storage&, Storage&, const WindowShape&) {
    return false;
}
bool maxpool_backward(const Storage&, const Storage&, Storage&, const WindowShape&) {
    return false;
}
bool permute(const Storage&, Storage&, const size_t*, const size_t*, size_t) {
    return false;
}
bool sum_axis(const Storage&, Storage&, size_t, size_t, size_t) {
    return false;
}
bool reduce_sum(const Storage&, double&) {
    return false;
}
bool reduce_sum_squares(const Storage&, double&) {
    return false;
}
bool scale_in_place(Storage&, float) {
    return false;
}
bool sgd_step(Storage&, const Storage&, Storage*, float, float, float) {
    return false;
}
bool adam_step(Storage&, const Storage&, Storage&, Storage&, float, float, float, float, float,
               float, float) {
    return false;
}
bool relu(const Storage&, Storage&) {
    return false;
}
bool relu_backward(const Storage&, const Storage&, Storage&) {
    return false;
}
bool accumulate_grad(Storage&, const Storage&, bool) {
    return false;
}
bool layernorm(const Storage&, const Storage&, const Storage&, Storage&, Storage&, Storage&, size_t,
               size_t, float) {
    return false;
}
bool layernorm_backward(const Storage&, const Storage&, const Storage&, const Storage&, Storage&,
                        Storage&, Storage&, size_t, size_t) {
    return false;
}
bool softmax(const Storage&, Storage&, size_t, size_t) {
    return false;
}
bool softmax_backward(const Storage&, const Storage&, Storage&, size_t, size_t) {
    return false;
}

}  // namespace ops

}  // namespace cuda
}  // namespace engine

#endif  // !ENGINE_CUDA
