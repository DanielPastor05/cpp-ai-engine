#include "engine/detail/storage.hpp"

#ifdef ENGINE_CUDA
#include "engine/cuda.hpp"
#endif

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <mutex>
#include <unordered_map>
#include <utility>
#include <vector>

namespace engine {
namespace {

// A free list of host buffers, keyed by exact element count.
//
// The measurement that made this necessary: on the largest tensor the MNIST
// model produces, (64, 16, 28, 28) or 3.21 MB, allocating and zero-filling the
// output buffer takes **0.575 ms**, against 0.044 ms to write the same number of
// bytes into a buffer that already exists. Twelve times over, and it is 74% of a
// relu and 64% of an addition. Every operation in this engine returns a new
// tensor, so every operation pays it.
//
// It is not bandwidth: over warm memory the same write runs at 72 GB/s, which is
// what this machine's DDR4 should do. It is 800 demand-zero page faults, one per
// 4 KB, on memory the allocator has just handed back from the OS -- and a
// training loop asks for the same shapes on every step, so it pays them again
// every step.
//
// What this deliberately does NOT do is skip the fill. That would be the CPU
// twin of device_write(), and it carries the same trap: correct only when the
// caller overwrites every element, silently wrong when it does not, and matmul
// accumulates into an output it assumes is zeroed. Recycling keeps the semantics
// identical -- the buffer is still filled, just over pages that are already
// resident -- and removes the part that was never doing anything.
//
// ponytail: one free list per exact count, and a cap. Sizes recur exactly in a
// training loop, so exact matching hits; a size-class allocator would serve
// irregular workloads better and is the upgrade if one ever turns up.
class BufferPool {
public:
    std::vector<float> take(size_t count) {
        if (count == 0) return {};
        const std::lock_guard<std::mutex> lock(mutex_);
        const auto it = free_.find(count);
        if (it == free_.end() || it->second.empty()) {
            ++fresh_;
            return {};
        }
        std::vector<float> buffer = std::move(it->second.back());
        it->second.pop_back();
        pooled_ -= count;
        ++recycled_;
        return buffer;
    }

    BufferPoolStats stats() {
        const std::lock_guard<std::mutex> lock(mutex_);
        BufferPoolStats out;
        out.fresh = fresh_;
        out.recycled = recycled_;
        out.bytes_held = pooled_ * sizeof(float);
        return out;
    }

    void give(std::vector<float> buffer) {
        const size_t count = buffer.size();
        if (count == 0) return;
        const std::lock_guard<std::mutex> lock(mutex_);
        // Over the cap the buffer is simply let go, which frees it the ordinary
        // way. Without this a program that allocates one enormous tensor and never
        // asks for that shape again would hold it for the rest of the run.
        if (pooled_ + count > cap_floats()) return;
        pooled_ += count;
        free_[count].push_back(std::move(buffer));
    }

    // How much the pool may hold, in bytes. 128 MB by default; override with
    // ENGINE_BUFFER_POOL_MB, and 0 turns the pool off entirely.
    //
    // The default is a stated trade rather than a tuned one. Measured on MNIST,
    // peak RSS against training time:
    //
    //     no pool   110 MB   25.3 s
    //      64 MB    168 MB   19.6 s
    //     128 MB    219 MB   18.8 s
    //     192 MB    247 MB   18.3 s
    //     256 MB    247 MB   18.2 s   <- the working set is already covered
    //
    // 128 MB takes 87% of the available speedup. Past it the curve is 0.5 s for
    // another 28 MB, and a caller who would rather have the memory back has the
    // variable. Doubling peak RSS to make training 1.35x faster is the same trade
    // PyTorch's caching allocator makes, and it should be as easy to refuse.
    static size_t cap_floats() {
        static const size_t cap = [] {
            const char* raw = std::getenv("ENGINE_BUFFER_POOL_MB");
            size_t megabytes = 128;
            if (raw != nullptr && raw[0] != '\0') {
                const long parsed = std::strtol(raw, nullptr, 10);
                if (parsed >= 0) megabytes = static_cast<size_t>(parsed);
            }
            return (megabytes << 20) / sizeof(float);
        }();
        return cap;
    }

private:
    std::mutex mutex_;
    std::unordered_map<size_t, std::vector<std::vector<float>>> free_;
    size_t pooled_ = 0;
    size_t fresh_ = 0;
    size_t recycled_ = 0;
};

// Leaked on purpose, and this is not a leak in the sense that matters.
//
// A function-local static would be destroyed at exit, and any Buffer outliving
// it -- a static Tensor, a thread still unwinding -- would hand its vector to a
// destroyed object. Never destroying the pool makes that impossible. The memory
// is returned to the OS when the process ends, which is the same moment the
// destructor would have run.
BufferPool& pool() {
    static BufferPool* instance = new BufferPool();
    return *instance;
}

}  // namespace

BufferPoolStats buffer_pool_stats() {
    return pool().stats();
}

Storage::Buffer::~Buffer() {
#ifdef ENGINE_CUDA
    if (device != nullptr) cuda::detail::device_free(device);
#endif
    // The host vector goes back to the free list rather than to the allocator, so
    // the next tensor of this shape reuses pages that are already resident. See
    // BufferPool above for the measurement that put it there.
    pool().give(std::move(host));
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
// engine â€” captured in lambdas, returned by value â€” and if that quietly aliased
// buffers, a backward_fn holding a captured input would see it change under it.
// Sharing has to be asked for by name: share().
Storage::Storage(const Storage& other) : buf_(std::make_shared<Buffer>()) {
    buf_->count = other.buf_->count;
    buf_->fill = other.buf_->fill;
    // An unmaterialised source has nothing to copy: the copy inherits the same
    // description and allocates when it is read, exactly as the original would.
    if (other.buf_->host.size() == other.buf_->count) buf_->host = other.host();
}

// Copy and move: the copy constructor already knows how to duplicate a buffer,
// including the case where the source has not been materialised, so this builds
// one and takes its buffer rather than repeating the logic.
Storage& Storage::operator=(const Storage& other) {
    if (this != &other) {
        Storage copy(other);
        buf_ = std::move(copy.buf_);
    }
    return *this;
}

Storage::~Storage() = default;

void Storage::materialise() const {
    if (buf_->host.size() != buf_->count) {
        std::vector<float> recycled = pool().take(buf_->count);
        if (recycled.empty()) {
            buf_->host.assign(buf_->count, buf_->fill);
        } else {
            buf_->host = std::move(recycled);
            std::fill(buf_->host.begin(), buf_->host.end(), buf_->fill);
        }
    }
    // The postcondition every caller relies on and none of them state: after
    // this, host.size() is the element count. The lazy mirror made those two
    // able to disagree, and the whole class of bug it can cause is a loop that
    // trusts `count` walking off the end of a vector that was never grown.
    assert(buf_->host.size() == buf_->count &&
           "Storage: the host mirror does not match the element count");
}

const std::vector<float>& Storage::host() const {
    // Callers index this with anything up to size(), which reads `count`.

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
    assert((buf_->host_valid || buf_->device_valid) &&
           "Storage: sync_host() with neither copy valid -- nothing to sync from");
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
    // The one invariant this class exists to hold, asserted where it can
    // actually break. sync_device() has just set device_valid, so dropping the
    // host copy leaves the device one -- unless sync_device() took its early
    // return on a zero-length buffer that never had a device pointer.
    assert((buf_->host_valid || buf_->device_valid) &&
           "Storage: neither copy is valid after device_mut()");
    return buf_->device;
}

float* Storage::device_write() {
    ensure_device_buffer();
    // A zero-length buffer gets no device allocation, so claiming the device
    // copy is the valid one would be a lie that the next host read believes.
    assert((buf_->count == 0 || buf_->device != nullptr) &&
           "Storage: device_write() with nothing allocated to write to");
    // Nothing is uploaded: the kernel is going to write the whole buffer. What was
    // on the host stops being valid as soon as it comes back.
    buf_->host_valid = false;
    buf_->device_valid = true;
    return buf_->device;
}

void Storage::revert_device_write() {
    // Only ever correct straight after a device_write() whose kernel did not
    // launch: the host copy is still whatever it was, and the device one holds
    // uninitialised memory. Calling it at any other time hands the caller a
    // stale host buffer and calls it current.
    assert(!buf_->host_valid && "Storage: revert_device_write() with a valid host copy");
    buf_->host_valid = true;
    buf_->device_valid = false;
}

#endif  // ENGINE_CUDA

}  // namespace engine
