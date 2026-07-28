#ifndef ENGINE_DETAIL_STORAGE_HPP
#define ENGINE_DETAIL_STORAGE_HPP

#include <cstddef>
#include <vector>

namespace engine {

// El búfer de un tensor, y de qué lado vive.
//
// Sin CUDA esto es exactamente un std::vector<float>: los miembros del
// dispositivo ni siquiera se declaran, así que la compilación de CPU no paga
// nada por existir el backend.
//
// Con CUDA mantiene además un espejo en el dispositivo y dos banderas de
// validez. Nadie copia nada hasta que alguien pide el lado que está obsoleto,
// de modo que una cadena de operaciones en GPU se queda en la GPU y sólo baja
// a host cuando el programa lee los valores.
//
// Ésta es la separación que la Fase 6 necesitaba **antes** del primer kernel.
// TensorImpl guardaba un std::vector<float>, que es host y punto; sin sacar el
// búfer a un tipo propio, cada operación habría acabado con ramas
// host/dispositivo repartidas por todo src/tensor.cpp y cada kernel habría
// pagado una ida y vuelta por PCIe.
//
// Invariante: al menos una de las dos copias es válida en todo momento.
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

    // ---- lado host ----
    // La versión mutable marca el espejo del dispositivo como obsoleto: si el
    // programa escribe en el búfer de host, lo que hubiera en la GPU deja de
    // valer.
    const std::vector<float>& host() const;
    std::vector<float>& host_mut();

    float& operator[](size_t i) { return host_mut()[i]; }
    const float& operator[](size_t i) const { return host()[i]; }
    float& at(size_t i) { return host_mut().at(i); }
    const float& at(size_t i) const { return host().at(i); }

    void assign(size_t count, float value);

    // ---- lado dispositivo ----
#ifdef ENGINE_CUDA
    // Sube si hace falta y devuelve el puntero de dispositivo (sólo lectura).
    const float* device() const;
    // Igual, pero además marca el host como obsoleto: el kernel va a escribir.
    float* device_mut();
    // Reserva sin subir nada. Para la salida de un kernel, que se escribe
    // entera: subir su contenido previo sería tráfico por PCIe tirado.
    float* device_write();

    // Deshace un device_write() cuyo kernel no llegó a lanzarse. Sin esto, el
    // camino de recuperación bajaría a host el búfer de dispositivo sin
    // inicializar y se cargaría el resultado que la CPU está a punto de
    // calcular —de forma especialmente traicionera en matmul, que acumula
    // sobre una salida que da por puesta a cero—.
    void revert_device_write();

    // True si la copia válida más reciente está en el dispositivo. Lo usan el
    // banco de pruebas y las decisiones de despacho, no la corrección.
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
