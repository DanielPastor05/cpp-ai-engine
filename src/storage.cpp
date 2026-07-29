#include "engine/detail/storage.hpp"

#ifdef ENGINE_CUDA
#include "engine/cuda.hpp"
#endif

#include <utility>

namespace engine {

Storage::Storage(size_t count, float fill) : host_(count, fill) {}

Storage::Storage(std::vector<float> values) : host_(std::move(values)) {}

// La copia se lleva sólo el lado host. Duplicar además el búfer del
// dispositivo reservaría memoria de GPU en cada copia de un tensor, que es lo
// último que conviene en el bucle de entrenamiento; el espejo se vuelve a
// crear en cuanto alguien pida el dispositivo.
Storage::Storage(const Storage& other) : host_(other.host()) {}

Storage& Storage::operator=(const Storage& other) {
    if (this != &other) {
#ifdef ENGINE_CUDA
        release_device();
        host_valid_ = true;
        device_valid_ = false;
#endif
        host_ = other.host();
    }
    return *this;
}

Storage::Storage(Storage&& other) noexcept : host_(std::move(other.host_)) {
#ifdef ENGINE_CUDA
    device_ = other.device_;
    host_valid_ = other.host_valid_;
    device_valid_ = other.device_valid_;
    other.device_ = nullptr;
    other.host_valid_ = true;
    other.device_valid_ = false;
#endif
}

Storage& Storage::operator=(Storage&& other) noexcept {
    if (this != &other) {
#ifdef ENGINE_CUDA
        release_device();
        device_ = other.device_;
        host_valid_ = other.host_valid_;
        device_valid_ = other.device_valid_;
        other.device_ = nullptr;
        other.host_valid_ = true;
        other.device_valid_ = false;
#endif
        host_ = std::move(other.host_);
    }
    return *this;
}

Storage::~Storage() {
#ifdef ENGINE_CUDA
    release_device();
#endif
}

const std::vector<float>& Storage::host() const {
#ifdef ENGINE_CUDA
    sync_host();
#endif
    return host_;
}

std::vector<float>& Storage::host_mut() {
#ifdef ENGINE_CUDA
    sync_host();
    // Quien escriba por esta referencia deja obsoleto lo que hubiera en la GPU.
    device_valid_ = false;
#endif
    return host_;
}

Storage Storage::clone() const {
#ifdef ENGINE_CUDA
    // Solo cuando el dispositivo es el unico que tiene el dato bueno. Si host
    // tambien vale, copiarlo es gratis y ademas no reserva memoria de GPU.
    if (device_valid_ && !host_valid_ && device_ != nullptr) {
        Storage copy;
        // El vector se dimensiona pero no se rellena: size() sale de el, y su
        // contenido esta marcado obsoleto hasta que alguien pida el lado host.
        copy.host_.resize(host_.size());
        copy.ensure_device_buffer();
        cuda::detail::copy_device_to_device(copy.device_, device_, host_.size());
        copy.host_valid_ = false;
        copy.device_valid_ = true;
        return copy;
    }
#endif
    return Storage(host_);
}

void Storage::assign(size_t count, float value) {
#ifdef ENGINE_CUDA
    if (count != host_.size()) release_device();
    host_valid_ = true;
    device_valid_ = false;
#endif
    host_.assign(count, value);
}

#ifdef ENGINE_CUDA

void Storage::release_device() {
    if (device_ != nullptr) {
        cuda::detail::device_free(device_);
        device_ = nullptr;
    }
    device_valid_ = false;
}

void Storage::ensure_device_buffer() const {
    if (device_ == nullptr && !host_.empty()) {
        device_ = cuda::detail::device_alloc(host_.size());
    }
}

void Storage::sync_host() const {
    if (host_valid_) return;
    if (device_ != nullptr && !host_.empty()) {
        cuda::detail::copy_to_host(host_.data(), device_, host_.size());
    }
    host_valid_ = true;
}

void Storage::sync_device() const {
    ensure_device_buffer();
    if (device_valid_) return;
    if (device_ != nullptr && !host_.empty()) {
        cuda::detail::copy_to_device(device_, host_.data(), host_.size());
    }
    device_valid_ = true;
}

const float* Storage::device() const {
    sync_device();
    return device_;
}

float* Storage::device_mut() {
    sync_device();
    host_valid_ = false;
    return device_;
}

float* Storage::device_write() {
    ensure_device_buffer();
    // No se sube nada: el kernel va a escribir el búfer entero. Lo que había
    // en host deja de valer en cuanto vuelva.
    host_valid_ = false;
    device_valid_ = true;
    return device_;
}

void Storage::revert_device_write() {
    host_valid_ = true;
    device_valid_ = false;
}

#endif // ENGINE_CUDA

} // namespace engine
