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

// No allocation here on purpose: see Buffer's comment. The values are described,
// not built, and materialise() builds them the first time anyone looks.
Storage::Storage(size_t count, float fill) : buf_(std::make_shared<Buffer>()) {
    buf_->count = count;
    buf_->fill = fill;
}

Storage::Storage(std::vector<float> values) : buf_(std::make_shared<Buffer>()) {
    buf_->count = values.size();
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
    buf_->count = other.buf_->count;
    buf_->fill = other.buf_->fill;
    // An unmaterialised source has nothing to copy: the copy inherits the same
    // description and allocates when it is read, exactly as the original would.
    if (other.buf_->host.size() == other.buf_->count) buf_->host = other.host();
}

Storage& Storage::operator=(const Storage& other) {
    if (this != &other) {
        const Storage copy(other);
        buf_ = copy.buf_;
    }
    return *this;
}

Storage::~Storage() = default;

void Storage::materialise() const {
    if (buf_->host.size() != buf_->count) buf_->host.assign(buf_->count, buf_->fill);
}

const std::vector<float>& Storage::host() const {
#ifdef ENGINE_CUDA
    sync_host();
#endif
    materialise();
    return buf_->host;
}

std::vector<float>& Storage::host_mut() {
#ifdef ENGINE_CUDA
    sync_host();
    // Anyone writing through this reference makes whatever is on the GPU stale.
    buf_->device_valid = false;
#endif
    materialise();
    return buf_->host;
}

Storage Storage::clone() const {
    Storage copy;
    copy.buf_->count = buf_->count;
    copy.buf_->fill = buf_->fill;
#ifdef ENGINE_CUDA
    // Only when the device is the sole holder of the good data. If host is valid
    // too, copying it is free and allocates no GPU memory either.
    if (buf_->device_valid && !buf_->host_valid && buf_->device != nullptr) {
        copy.ensure_device_buffer();
        cuda::detail::copy_device_to_device(copy.buf_->device, buf_->device, buf_->count);
        copy.buf_->host_valid = false;
        copy.buf_->device_valid = true;
        return copy;
    }
#endif
    if (buf_->host.size() == buf_->count) copy.buf_->host = host();
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
    if (count != buf_->count && buf_->device != nullptr) {
        cuda::detail::device_free(buf_->device);
        buf_->device = nullptr;
    }
    buf_->host_valid = true;
    buf_->device_valid = false;
#endif
    // Lazy for the same reason as the constructor: zero_grad() assigns every
    // gradient in the model and on the device path a kernel overwrites each one.
    buf_->count = count;
    buf_->fill = value;
    buf_->host.clear();
}

#ifdef ENGINE_CUDA

// The whole point of splitting count out of the host vector: a tensor that is
// only ever written and read by kernels never allocates a host mirror at all.
void Storage::ensure_device_buffer() const {
    if (buf_->device == nullptr && buf_->count != 0) {
        buf_->device = cuda::detail::device_alloc(buf_->count);
    }
}

void Storage::sync_host() const {
    if (buf_->host_valid) return;
    if (buf_->device != nullptr && buf_->count != 0) {
        // resize and not materialise(): the download is about to overwrite every
        // value, so filling them first would be the waste this change removes.
        buf_->host.resize(buf_->count);
        cuda::detail::copy_to_host(buf_->host.data(), buf_->device, buf_->count);
    }
    buf_->host_valid = true;
}

void Storage::sync_device() const {
    ensure_device_buffer();
    if (buf_->device_valid) return;
    if (buf_->device != nullptr && buf_->count != 0) {
        materialise();
        cuda::detail::copy_to_device(buf_->device, buf_->host.data(), buf_->count);
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
