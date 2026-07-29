#ifndef ENGINE_DETAIL_STORAGE_HPP
#define ENGINE_DETAIL_STORAGE_HPP

#include <cstddef>
#include <vector>

namespace engine {

// A tensor's buffer, and which side of the bus it lives on.
//
// Without CUDA this is exactly a std::vector<float>: the device members are not
// even declared, so a CPU build pays nothing for the backend existing at all.
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
    Storage() = default;
    Storage(size_t count, float fill);
    explicit Storage(std::vector<float> values);

    Storage(const Storage& other);
    Storage& operator=(const Storage& other);
    Storage(Storage&& other) noexcept;
    Storage& operator=(Storage&& other) noexcept;
    ~Storage();

    size_t size() const { return host_.size(); }
    bool empty() const { return host_.empty(); }

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

    // A copy with the same values that **keeps whichever side they live on**:
    // if the data is only on the device, it is duplicated there with an
    // internal memcpy instead of being pulled down and pushed back up.
    //
    // The copy constructor cannot do this: it runs when tensors are copied
    // around, and allocating GPU memory on every tensor copy is the last thing
    // a training loop needs. Here the situation is the opposite — reshape is
    // going to have both copies live anyway — and it saves the round trip.
    Storage clone() const;

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
    bool resident_on_device() const { return device_valid_; }
#endif

private:
#ifdef ENGINE_CUDA
    void ensure_device_buffer() const;
    void sync_host() const;
    void sync_device() const;
    void release_device();

    mutable float* device_ = nullptr;
    mutable bool host_valid_ = true;
    mutable bool device_valid_ = false;
#endif
    mutable std::vector<float> host_;
};

} // namespace engine

#endif // ENGINE_DETAIL_STORAGE_HPP
