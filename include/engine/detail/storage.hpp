#ifndef ENGINE_DETAIL_STORAGE_HPP
#define ENGINE_DETAIL_STORAGE_HPP

#include <cstddef>
#include <memory>
#include <vector>

namespace engine {

// A tensor's buffer, and which side of the bus it lives on.
//
// Without CUDA this is a std::vector<float> behind a shared handle: the CPU
// build pays one small allocation for the handle and nothing at all for the
// backend existing.
//
// With CUDA it also keeps a device mirror and two validity flags. Nothing is
// copied until somebody asks for the side that has gone stale, so a chain of
// GPU operations stays on the GPU and only comes back down when the program
// reads the values.
//
// This is the separation Phase 6 needed **before** its first kernel. TensorImpl
// used to hold a std::vector<float>, which is host and nothing else; without
// lifting the buffer into a type of its own, every operation would have ended
// up with host/device branches spread through src/tensor.cpp and every kernel
// would have paid a round trip over PCIe.
//
// Invariant: at least one of the two copies is valid at all times.
class Storage {
public:
    Storage();
    Storage(size_t count, float fill);
    explicit Storage(std::vector<float> values);

    Storage(const Storage& other);
    Storage& operator=(const Storage& other);
    Storage(Storage&& other) noexcept = default;
    Storage& operator=(Storage&& other) noexcept = default;
    ~Storage();

    size_t size() const { return buf_->host.size(); }
    bool empty() const { return buf_->host.empty(); }

    // ---- host side ----
    // The mutable version marks the device mirror stale: if the program writes
    // to the host buffer, whatever is on the GPU stops being valid.
    const std::vector<float>& host() const;
    std::vector<float>& host_mut();

    float& operator[](size_t i) { return host_mut()[i]; }
    const float& operator[](size_t i) const { return host()[i]; }
    float& at(size_t i) { return host_mut().at(i); }
    const float& at(size_t i) const { return host().at(i); }

    void assign(size_t count, float value);

    // An independent copy that **keeps whichever side the values live on**: if
    // the data is only on the device, it is duplicated there with an internal
    // memcpy instead of being pulled down and pushed back up.
    //
    // The copy constructor cannot do this: it runs when tensors are copied
    // around, and allocating GPU memory on every tensor copy is the last thing
    // a training loop needs. Here the situation is the opposite — the caller is
    // going to have both copies live anyway — and it saves the round trip.
    Storage clone() const;

    // A second handle on the **same** buffer: both see each other's writes, and
    // both see the same host/device residency, because there is only one of
    // each. This is what makes reshape a view rather than a copy.
    //
    // Sharing is the dangerous one of the three, so it is a named method rather
    // than a copy constructor: the caller has to mean it. `reshape` means it.
    Storage share() const;

    // True if two handles refer to the same buffer. For tests and assertions.
    bool shares_with(const Storage& other) const { return buf_ == other.buf_; }

    // ---- device side ----
#ifdef ENGINE_CUDA
    // Uploads if needed and returns the device pointer (read only).
    const float* device() const;
    // The same, but also marks the host stale: the kernel is going to write.
    float* device_mut();
    // Allocates without uploading. For a kernel's output, which is written in
    // full: uploading its previous contents would be PCIe traffic thrown away.
    float* device_write();

    // Undoes a device_write() whose kernel never launched. Without this, the
    // recovery path would pull an uninitialised device buffer down to host and
    // destroy the result the CPU is about to compute — especially treacherous
    // in matmul, which accumulates into an output it assumes is zeroed.
    void revert_device_write();

    // True if the most recent valid copy is on the device. Used by the test
    // suite and by dispatch decisions, never by correctness.
    bool resident_on_device() const { return buf_->device_valid; }
#endif

private:
    // Everything that a view has to see through its sibling lives here: the
    // values, the device mirror and the two flags. Splitting any of it out
    // would let two handles disagree about where the good copy is, which is the
    // one thing the invariant exists to prevent.
    struct Buffer {
        std::vector<float> host;
#ifdef ENGINE_CUDA
        float* device = nullptr;
        bool host_valid = true;
        bool device_valid = false;
#endif
        ~Buffer();
    };

#ifdef ENGINE_CUDA
    void ensure_device_buffer() const;
    void sync_host() const;
    void sync_device() const;
#endif

    std::shared_ptr<Buffer> buf_;
};

}  // namespace engine

#endif  // ENGINE_DETAIL_STORAGE_HPP
