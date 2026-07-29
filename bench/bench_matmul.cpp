// Banco de pruebas del producto de matrices en GPU.
//
// Existe aparte de bench.cpp por una razón práctica: para perfilar con Nsight
// Compute hace falta un ejecutable que lance **un** kernel sobre **una** forma.
// Pasarle el banco completo a ncu significa esperar a que perfile decenas de
// lanzamientos distintos para leer uno.
//
//   bench_matmul                                    # barre todo y saca la tabla
//   bench_matmul --kernel=register --size=2048      # una variante, una forma
//   ncu --set full -o perfil bench_matmul --kernel=register --size=2048 --iters=10
//
// Las cifras que imprime son las que van a docs/CUDA.md. No hay ninguna
// escrita a mano en la documentación.

#include "engine/tensor.hpp"
#include "engine/random.hpp"
#include "engine/cuda.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using engine::Tensor;
namespace cuda = engine::cuda;

namespace {

struct Options {
    cuda::MatmulKernel kernel = cuda::MatmulKernel::Auto;
    bool sweep = true;  // sin argumentos, se barre todo
    size_t size = 2048;
    size_t iters = 0;  // 0 = por tiempo, no por repeticiones
};

bool parse_kernel(const std::string& value, cuda::MatmulKernel& out) {
    if (value == "auto") {
        out = cuda::MatmulKernel::Auto;
        return true;
    }
    if (value == "naive") {
        out = cuda::MatmulKernel::Naive;
        return true;
    }
    if (value == "tiled") {
        out = cuda::MatmulKernel::Tiled;
        return true;
    }
    if (value == "register") {
        out = cuda::MatmulKernel::RegisterTiled;
        return true;
    }
    if (value == "vectorized") {
        out = cuda::MatmulKernel::Vectorized;
        return true;
    }
    return false;
}

bool parse_args(int argc, char** argv, Options& opts) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg.rfind("--kernel=", 0) == 0) {
            if (!parse_kernel(arg.substr(9), opts.kernel)) {
                printf("Kernel desconocido: %s\n", arg.substr(9).c_str());
                return false;
            }
            opts.sweep = false;
        } else if (arg.rfind("--size=", 0) == 0) {
            opts.size = static_cast<size_t>(std::stoul(arg.substr(7)));
            opts.sweep = false;
        } else if (arg.rfind("--iters=", 0) == 0) {
            opts.iters = static_cast<size_t>(std::stoul(arg.substr(8)));
        } else {
            printf(
                "Uso: bench_matmul [--kernel=auto|naive|tiled|register|vectorized]\n"
                "                  [--size=N] [--iters=N]\n");
            return false;
        }
    }
    return true;
}

// Devuelve el tiempo medio por producto, en segundos.
//
// Los operandos se dejan residentes en el dispositivo antes de empezar: aquí
// se mide el kernel, no el PCIe. El coste de las transferencias se mide en
// bench.cpp, y por separado a propósito.
double time_matmul(const Tensor& A, const Tensor& B, size_t iters, double min_seconds) {
    { Tensor warm = A.matmul(B); }  // sube los operandos y compila el camino
    cuda::synchronize();

    const auto start = std::chrono::steady_clock::now();
    size_t done = 0;
    double elapsed = 0.0;
    while (iters > 0 ? (done < iters) : (elapsed < min_seconds)) {
        Tensor C = A.matmul(B);
        ++done;
        if (iters == 0) {
            cuda::synchronize();
            elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        }
    }
    cuda::synchronize();
    elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    return elapsed / static_cast<double>(done);
}

void run_one(cuda::MatmulKernel kernel, size_t n, size_t iters, double peak) {
    cuda::set_matmul_kernel(kernel);

    Tensor A = Tensor::randn({n, n});
    Tensor B = Tensor::randn({n, n});

    const double seconds = time_matmul(A, B, iters, 0.4);
    const double gflops = 2.0 * (double)n * n * n / seconds / 1e9;
    const double share = (peak > 0.0) ? (gflops / peak * 100.0) : 0.0;

    printf("  %-14s %10.3f ms %12.1f %11.1f%%\n", cuda::matmul_kernel_name(kernel),
           seconds * 1000.0, gflops, share);
}

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    if (!parse_args(argc, argv, opts)) return 1;

    engine::manual_seed(1);

    if (!cuda::available()) {
        printf(
            "Compilado sin CUDA o sin dispositivo utilizable.\n"
            "  cmake -B build-cuda -S . -DENGINE_CUDA=ON\n");
        return 0;
    }

    const cuda::DeviceInfo info = cuda::device_info();
    const double peak = cuda::peak_fp32_gflops();

    printf("Dispositivo: %s (cc %d.%d, %d SM, %zu MiB)\n", info.name.c_str(), info.compute_major,
           info.compute_minor, info.multiprocessors, info.total_memory >> 20);
    printf("Pico teorico: %.0f GFLOP/s en fp32, %.0f GB/s de ancho de banda\n", peak,
           cuda::peak_bandwidth_gbs());

    // El punto de inflexion del modelo roofline: por debajo de esta intensidad
    // aritmetica manda la memoria, por encima manda el calculo. Un matmul de
    // NxNxN mueve 3*N^2 valores para hacer 2*N^3 operaciones, o sea N/6
    // FLOP/byte: cualquier N por encima de unos pocos cientos esta de lleno en
    // la region limitada por calculo, y por eso el trabajo va en la intensidad
    // aritmetica del kernel y no en las transferencias.
    const double ridge = peak / cuda::peak_bandwidth_gbs();
    printf("Punto de inflexion del roofline: %.1f FLOP/byte\n\n", ridge);

    // Sin umbral: aqui se mide siempre en GPU, incluso donde no compensaria.
    cuda::set_thresholds(0, 0);

    if (!opts.sweep) {
        const double intensity = (double)opts.size / 6.0;
        printf("Forma %zux%zux%zu, intensidad aritmetica %.0f FLOP/byte (%s)\n", opts.size,
               opts.size, opts.size, intensity,
               intensity > ridge ? "limitado por calculo" : "limitado por memoria");
        printf("  %-14s %10s %12s %11s\n", "kernel", "tiempo", "GFLOP/s", "% del pico");
        run_one(opts.kernel, opts.size, opts.iters, peak);
        printf("\n");
        return 0;
    }

    const cuda::MatmulKernel variants[] = {
        cuda::MatmulKernel::Naive,
        cuda::MatmulKernel::Tiled,
        cuda::MatmulKernel::RegisterTiled,
        cuda::MatmulKernel::Vectorized,
    };

    for (size_t n : {size_t{512}, size_t{1024}, size_t{2048}, size_t{4096}}) {
        printf("=== %zux%zux%zu (%.0f FLOP/byte) ===\n", n, n, n, (double)n / 6.0);
        printf("  %-14s %10s %12s %11s\n", "kernel", "tiempo", "GFLOP/s", "% del pico");
        for (cuda::MatmulKernel v : variants) {
            // El kernel ingenuo sobre 4096^3 tarda demasiado para lo que aporta.
            if (v == cuda::MatmulKernel::Naive && n > 2048) continue;
            run_one(v, n, opts.iters, peak);
        }
        printf("\n");
    }

    printf(
        "No se compara contra cuBLAS a proposito: el objetivo del proyecto es\n"
        "implementar el kernel, no llamarlo. La referencia util es el pico\n"
        "teorico de la tarjeta, que es la columna de la derecha.\n\n");

    cuda::set_matmul_kernel(cuda::MatmulKernel::Auto);
    return 0;
}
