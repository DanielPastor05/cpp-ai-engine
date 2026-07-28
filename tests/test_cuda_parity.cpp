// Paridad CPU / GPU.
//
// Cada caso calcula la misma expresión dos veces sobre exactamente los mismos
// datos —una con el backend apagado y otra con él encendido— y compara. Es la
// única forma de comprobar un kernel que sirve de algo: los kernels no fallan
// devolviendo un error, fallan devolviendo números plausibles.
//
// La comparación es con tolerancia y no exacta, y eso no es una concesión. El
// compilador de dispositivo funde multiplicación y suma en una FMA, que
// redondea una sola vez donde la CPU redondea dos; el resultado difiere en el
// último bit y va acumulándose con K. Exigir igualdad bit a bit entre CPU y GPU
// sería exigir que la GPU calcule peor.
//
// Sin ENGINE_CUDA el fichero se compila a una nota de que no hay nada que
// comprobar, para que la suite siga siendo una sola y CI no necesite GPU.

#include "test_support.hpp"

#include "engine/cuda.hpp"

#ifdef ENGINE_CUDA

#include <algorithm>
#include <cmath>

namespace {

namespace cuda = engine::cuda;

// Error relativo al tamaño del valor esperado: en un matmul con K grande los
// valores crecen como sqrt(K), y un umbral absoluto acabaría midiendo la
// magnitud de los datos en lugar de la calidad del kernel.
float max_relative_error(const std::vector<float>& got, const std::vector<float>& want) {
    if (got.size() != want.size()) return 1e30f;
    float worst = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float scale = std::max(1.0f, std::fabs(want[i]));
        worst = std::max(worst, std::fabs(got[i] - want[i]) / scale);
    }
    return worst;
}

// Evalúa la misma función con el backend apagado y encendido, y compara.
void compare(const std::string& what, const std::function<Tensor()>& compute,
             float tol = 1e-5f) {
    cuda::set_enabled(false);
    const std::vector<float> on_cpu = compute().data();

    cuda::set_enabled(true);
    const std::vector<float> on_gpu = compute().data();

    const float error = max_relative_error(on_gpu, on_cpu);
    ++testing::g_checks;
    if (error <= tol) {
        std::cout << "  [ ok ] " << what << " (error relativo maximo " << std::scientific
                  << std::setprecision(2) << error << std::defaultfloat << ")\n";
    } else {
        ++testing::g_failures;
        std::cout << "  [FAIL] " << what << " (error relativo maximo " << error
                  << " > " << tol << ")\n";
    }
}

} // namespace

void run_cuda_parity_tests() {
    testing::section("Paridad CPU / GPU (Fase 6)");

    if (!cuda::available()) {
        std::cout << "  (compilado con CUDA, pero no hay dispositivo: se omite)\n";
        return;
    }

    const cuda::DeviceInfo info = cuda::device_info();
    std::cout << "  Dispositivo: " << info.name << " (cc " << info.compute_major << "."
              << info.compute_minor << ", " << info.multiprocessors << " SM, "
              << (info.total_memory >> 20) << " MiB)\n";

    // Los umbrales normales mandarían a la CPU todo lo que hay aquí, que es
    // justo lo contrario de lo que quiere una prueba de paridad: interesa
    // ejercitar formas pequeñas y con restos, donde los bordes de las teselas
    // son quienes fallan. Se ponen a cero durante la prueba y se restauran al
    // final.
    const size_t saved_flops = cuda::min_matmul_flops();
    const size_t saved_elements = cuda::min_elementwise_elements();
    cuda::set_thresholds(0, 0);

    engine::manual_seed(20260728);

    // --- producto de matrices ---
    {
        // Tamaños deliberadamente no múltiplos de 32, el lado de la tesela:
        // si el relleno de los bordes estuviera mal, sólo se vería aquí.
        struct Case { size_t M, K, N; };
        const Case cases[] = {{1, 1, 1}, {17, 23, 31}, {32, 32, 32}, {33, 65, 129}, {128, 256, 64}};
        for (const Case& c : cases) {
            Tensor A = Tensor::randn({c.M, c.K});
            Tensor B = Tensor::randn({c.K, c.N});
            compare("matmul " + std::to_string(c.M) + "x" + std::to_string(c.K) + "x" +
                        std::to_string(c.N),
                    [&] { return A.matmul(B); });
        }

        Tensor QB = Tensor::randn({4, 3, 17, 23});
        Tensor KB = Tensor::randn({4, 3, 23, 11});
        compare("matmul por lotes (4,3,17,23) x (4,3,23,11)", [&] { return QB.matmul(KB); });

        // Operando 2D compartido con todo el lote: comprueba el paso 0.
        Tensor X = Tensor::randn({8, 12, 40});
        Tensor W = Tensor::randn({40, 20});
        compare("matmul con operando compartido (8,12,40) x (40,20)", [&] { return X.matmul(W); });
    }

    // --- operaciones elemento a elemento ---
    {
        Tensor A = Tensor::randn({64, 40});
        Tensor B = Tensor::randn({64, 40});
        compare("suma", [&] { return A + B; });
        compare("resta", [&] { return A - B; });
        compare("producto", [&] { return A * B; });

        // El divisor se aleja del cero: dividir por un valor casi nulo hace que
        // el resultado dependa del último bit del denominador, y eso mediría el
        // dato, no el kernel.
        Tensor D = Tensor::rand({64, 40}, 1.0f, 2.0f);
        compare("division", [&] { return A / D; });

        // Difusión: sesgo de una capa densa y codificación posicional.
        Tensor bias = Tensor::randn({40});
        compare("suma con difusion (64,40) + (40)", [&] { return A + bias; });

        Tensor seq = Tensor::randn({6, 9, 16});
        Tensor pe = Tensor::randn({9, 16});
        compare("suma con difusion (6,9,16) + (9,16)", [&] { return seq + pe; });
        compare("producto con difusion (6,9,16) * (9,16)", [&] { return seq * pe; });
    }

    // --- activaciones ---
    {
        Tensor X = Tensor::randn({48, 33});
        compare("relu", [&] { return X.relu(); });

        // cols no múltiplo del tamaño de bloque de la reducción, para que el
        // bucle con paso de bloque tenga que dar más de una vuelta.
        Tensor S = Tensor::randn({20, 300});
        compare("softmax sobre el ultimo eje (20,300)", [&] { return S.softmax(); });

        Tensor A4 = Tensor::randn({3, 4, 7, 19});
        compare("softmax sobre (3,4,7,19)", [&] { return A4.softmax(); });
    }

    // --- gradientes ---
    {
        Tensor X = Tensor::randn({40, 24}, 0.0f, 1.0f, true);
        compare("gradiente de relu", [&] {
            X.zero_grad();
            Tensor y = X.relu();
            (y * y).sum().backward();
            return X.grad();
        });

        Tensor S = Tensor::randn({16, 50}, 0.0f, 1.0f, true);
        compare("gradiente de softmax", [&] {
            S.zero_grad();
            Tensor y = S.softmax();
            // Un gradiente aguas arriba no uniforme: con todos los pesos
            // iguales, un error de indexación se cancelaría solo.
            Tensor w = Tensor::zeros({16, 50});
            for (size_t i = 0; i < w.size(); ++i) w.data()[i] = 0.1f * static_cast<float>(i % 7);
            (y * w).sum().backward();
            return S.grad();
        });

        Tensor A = Tensor::randn({30, 40}, 0.0f, 1.0f, true);
        Tensor B = Tensor::randn({40, 25}, 0.0f, 1.0f, true);
        compare("gradiente de matmul respecto de A", [&] {
            A.zero_grad();
            B.zero_grad();
            A.matmul(B).sum().backward();
            return A.grad();
        });
    }

    // --- una red entera ---
    //
    // Las pruebas por operación pueden pasar y el modelo dar otra cosa: basta
    // con que una operación deje un tensor en el lado equivocado y otra lo lea
    // sin sincronizar. Este caso encadena atención, normalización, dos capas
    // densas y su paso hacia atrás, que es donde saldría.
    {
        engine::manual_seed(7);
        nn::TransformerBlock block(32, 4, 64);
        Tensor tokens = Tensor::randn({4, 12, 32}, 0.0f, 1.0f, true);

        compare("TransformerBlock: salida", [&] { return block(tokens); }, 1e-4f);

        compare("TransformerBlock: gradiente de la entrada", [&] {
            block.zero_grad();
            tokens.zero_grad();
            block(tokens).sum().backward();
            return tokens.grad();
        }, 1e-4f);

        compare("TransformerBlock: gradiente del primer parametro", [&] {
            block.zero_grad();
            tokens.zero_grad();
            block(tokens).sum().backward();
            return block.parameters()[0].grad();
        }, 1e-4f);
    }

    // --- contabilidad de transferencias ---
    //
    // No comprueba un número, comprueba el modelo de residencia: un tensor
    // calculado en la GPU no baja a host hasta que alguien lee sus valores.
    {
        cuda::set_enabled(true);
        Tensor A = Tensor::randn({64, 64});
        Tensor B = Tensor::randn({64, 64});
        A.data();  // fuerza que ambos estén en host antes de empezar a contar
        B.data();

        cuda::reset_transfer_stats();
        Tensor C = A.matmul(B);
        const cuda::TransferStats after_kernel = cuda::transfer_stats();
        testing::check(after_kernel.to_device_count == 2,
                       "matmul sube los dos operandos y nada mas");
        testing::check(after_kernel.to_host_count == 0,
                       "el resultado se queda en el dispositivo hasta que se lee");

        volatile float first = C.data()[0];
        (void)first;
        const cuda::TransferStats after_read = cuda::transfer_stats();
        testing::check(after_read.to_host_count == 1,
                       "leer el resultado provoca exactamente una bajada");

        // Y una cadena de operaciones no vuelve a subir lo que ya está arriba.
        cuda::reset_transfer_stats();
        Tensor D = A.matmul(B).relu();
        (void)D.data()[0];
        const cuda::TransferStats chained = cuda::transfer_stats();
        testing::check(chained.to_device_count == 0,
                       "una segunda operacion no resube operandos ya residentes");
    }

    cuda::set_thresholds(saved_flops, saved_elements);
    cuda::set_enabled(true);
}

#else // !ENGINE_CUDA

void run_cuda_parity_tests() {
    testing::section("Paridad CPU / GPU (Fase 6)");
    std::cout << "  (compilado sin CUDA: -DENGINE_CUDA=ON para incluir estas pruebas)\n";
}

#endif // ENGINE_CUDA
