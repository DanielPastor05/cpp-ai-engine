#include "engine/detail/storage.hpp"

#ifdef ENGINE_CUDA
#include "engine/cuda.hpp"
#endif

#include <utility>

namespace engine {

Storage::Buffer::~Buffer() {
#ifdef ENGINE_CUDA
    if (device != nullptr) cuda::detail::device_free(device);
#endif
}

Storage::Storage() : buf_(std::make_shared<Buffer>()) {}

Storage::Storage(size_t count, float fill) : buf_(std::make_shared<Buffer>()) {
    buf_->host.assign(count, fill);
}

Storage::Storage(std::vector<float> values) : buf_(std::make_shared<Buffer>()) {
    buf_->host = std::move(values);
}

// A copy takes only the host side, into a buffer of its own. Duplicating the
// device buffer as well would allocate GPU memory on every tensor copy, which is
// the last thing a training loop needs; the mirror is recreated as soon as
// somebody asks for the device.
//
// It is a copy and not a share, deliberately. Tensors get copied all over the
// engine — captured in lambdas, returned by value — and if that quietly aliased
// buffers, a backward_fn holding a captured input would see it change under it.
// Sharing has to be asked for by name: share().
Storage::Storage(const Storage& other) : buf_(std::make_shared<Buffer>()) {
    buf_->host = other.host();
}

Storage& Storage::operator=(const Storage& other) {
    if (this != &other) {
        buf_ = std::make_shared<Buffer>();
        buf_->host = other.host();
    }
    return *this;
}

Storage::~Storage() = default;

const std::vector<float>& Storage::host() const {
#ifdef ENGINE_CUDA
    sync_host();
#endif
    return buf_->host;
}

std::vector<float>& Storage::host_mut() {
#ifdef ENGINE_CUDA
    sync_host();
    // Anyone writing through this reference makes whatever is on the GPU stale.
    buf_->device_valid = false;
#endif
    return buf_->host;
}

Storage Storage::clone() const {
    Storage copy;
#ifdef ENGINE_CUDA
    // Only when the device is the sole holder of the good data. If host is valid
    // too, copying it is free and allocates no GPU memory either.
    if (buf_->device_valid && !buf_->host_valid && buf_->device != nullptr) {
        // The vector is sized but not filled: size() comes from it, and its
        // contents are marked stale until somebody asks for the host side.
        copy.buf_->host.resize(buf_->host.size());
        copy.ensure_device_buffer();
        cuda::detail::copy_device_to_device(copy.buf_->device, buf_->device, buf_->host.size());
        copy.buf_->host_valid = false;
        copy.buf_->device_valid = true;
        return copy;
    }
#endif
    copy.buf_->host = host();
    return copy;
}

Storage Storage::share() const {
    Storage view;
    view.buf_ = buf_;
    return view;
}

void Storage::assign(size_t count, float value) {
#ifdef ENGINE_CUDA
    // A resize invalidates the device buffer's size, so it goes rather than
    // being reused at the wrong length.
    if (count != buf_->host.size() && buf_->device != nullptr) {
        cuda::detail::device_free(buf_->device);
        buf_->device = nullptr;
    }
    buf_->host_valid = true;
    buf_->device_valid = false;
#endif
    buf_->host.assign(count, value);
}

#ifdef ENGINE_CUDA

void Storage::ensure_device_buffer() const {
    if (buf_->device == nullptr && !buf_->host.empty()) {
        buf_->device = cuda::detail::device_alloc(buf_->host.size());
    }
}

void Storage::sync_host() const {
    if (buf_->host_valid) return;
    if (buf_->device != nullptr && !buf_->host.empty()) {
        cuda::detail::copy_to_host(buf_->host.data(), buf_->device, buf_->host.size());
    }
    buf_->host_valid = true;
}

void Storage::sync_device() const {
    ensure_device_buffer();
    if (buf_->device_valid) return;
    if (buf_->device != nullptr && !buf_->host.empty()) {
        cuda::detail::copy_to_device(buf_->device, buf_->host.data(), buf_->host.size());
    }
    buf_->device_valid = true;
}

const float* Storage::device() const {
    sync_device();
    return buf_->device;
}

float* Storage::device_mut() {
    sync_device();
    buf_->host_valid = false;
    return buf_->device;
}

float* Storage::device_write() {
    ensure_device_buffer();
    // Nothing is uploaded: the kernel is going to write the whole buffer. What was
    // on the host stops being valid as soon as it comes back.
    buf_->host_valid = false;
    buf_->device_valid = true;
    return buf_->device;
}

void Storage::revert_device_write() {
    buf_->host_valid = true;
    buf_->device_valid = false;
}

#endif  // ENGINE_CUDA

}  // namespace engine
