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

// Traduce un índice plano a coordenadas, para que un fallo diga «fila 128,
// columna 0» en vez de «elemento 16384».
std::string position_of(size_t flat, const std::vector<size_t>& shape) {
    if (shape.empty()) return "(escalar)";
    std::vector<size_t> coords(shape.size());
    for (size_t axis = shape.size(); axis-- > 0;) {
        coords[axis] = (shape[axis] == 0) ? 0 : flat % shape[axis];
        flat = (shape[axis] == 0) ? 0 : flat / shape[axis];
    }
    std::string out = "(";
    for (size_t i = 0; i < coords.size(); ++i) {
        out += std::to_string(coords[i]);
        if (i + 1 < coords.size()) out += ", ";
    }
    return out + ")";
}

// Evalúa la misma función con el backend apagado y encendido, y compara.
//
// Cuando falla, imprime bastante más que el error máximo. El motivo es
// práctico: estos kernels no se pueden ejecutar en el entorno donde se
// escriben, así que cada iteración de depuración cuesta una ida y vuelta
// entera hasta una máquina con GPU. El mensaje tiene que llevar de una vez
// todo lo necesario para localizar el fallo.
//
// El **reparto** de las diferencias es más informativo que su tamaño:
//   - unas pocas al final  -> tratamiento de bordes
//   - todas                -> la fórmula, no los índices
//   - a intervalos regulares -> un paso o una transposición mal puestos
void compare(const std::string& what, const std::function<Tensor()>& compute,
             float tol = 1e-5f) {
    cuda::set_enabled(false);
    const Tensor on_cpu = compute();
    const std::vector<float> want = on_cpu.data();
    const std::vector<size_t> shape = on_cpu.shape();

    cuda::set_enabled(true);
    const size_t launched_before = cuda::kernels_launched();
    const std::vector<float> got = compute().data();
    const size_t launched = cuda::kernels_launched() - launched_before;

    ++testing::g_checks;

    // Sin esto la prueba es una trampa. El motor cae al camino de CPU cuando un
    // kernel no se puede lanzar —correcto en produccion—, de modo que la
    // comparacion seria CPU contra CPU y pasaria con error exactamente cero sin
    // haber tocado el dispositivo. Paso de verdad: toda la seccion salia en
    // verde con 0.00e+00 mientras ningun kernel llegaba a ejecutarse.
    if (launched == 0) {
        ++testing::g_failures;
        std::cout << "  [FAIL] " << what
                  << " (no se lanzo ningun kernel: se comparo CPU contra CPU)\n";
        return;
    }

    if (got.size() != want.size()) {
        ++testing::g_failures;
        std::cout << "  [FAIL] " << what << " (tamanos distintos: " << got.size()
                  << " frente a " << want.size() << ")\n";
        return;
    }

    size_t differing = 0;
    size_t first = want.size();
    size_t last = 0;
    size_t worst_at = 0;
    float worst = 0.0f;

    for (size_t i = 0; i < want.size(); ++i) {
        const float scale = std::max(1.0f, std::fabs(want[i]));
        const float error = std::fabs(got[i] - want[i]) / scale;
        if (error > worst) { worst = error; worst_at = i; }
        if (error > tol) {
            ++differing;
            if (first == want.size()) first = i;
            last = i;
        }
    }

    if (differing == 0) {
        std::cout << "  [ ok ] " << what << " (error relativo maximo " << std::scientific
                  << std::setprecision(2) << worst << std::defaultfloat << ")\n";
        return;
    }

    ++testing::g_failures;
    std::cout << "  [FAIL] " << what << "\n"
              << "         error relativo maximo " << std::scientific << std::setprecision(3)
              << worst << " > " << tol << std::defaultfloat << "\n"
              << "         difieren " << differing << " de " << want.size()
              << " elementos, del " << first << " al " << last << "\n"
              << "         el peor en " << worst_at << " = " << position_of(worst_at, shape)
              << ": esperado " << want[worst_at] << ", obtenido " << got[worst_at] << "\n";

    if (differing == want.size()) {
        std::cout << "         difieren todos: apunta a la formula, no a los indices\n";
    } else if (first > 0 && last + 1 == want.size()) {
        std::cout << "         difiere la cola: apunta al tratamiento de bordes\n";
    } else if (differing * 4 < want.size()) {
        std::cout << "         difiere una minoria dispersa: apunta a un paso o una"
                     " transposicion\n";
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
    const int rt = cuda::runtime_version();
    const int drv = cuda::driver_version();
    std::cout << "  Dispositivo: " << info.name << " (cc " << info.compute_major << "."
              << info.compute_minor << ", " << info.multiprocessors << " SM, "
              << (info.total_memory >> 20) << " MiB)\n"
              << "  Compilado con CUDA " << rt / 1000 << "." << (rt % 1000) / 10
              << ", driver CUDA " << drv / 1000 << "." << (drv % 1000) / 10 << "\n";

    cuda::reset_kernel_counters();

    // Los umbrales normales mandarían a la CPU todo lo que hay aquí, que es
    // justo lo contrario de lo que quiere una prueba de paridad: interesa
    // ejercitar formas pequeñas y con restos, donde los bordes de las teselas
    // son quienes fallan. Se ponen a cero durante la prueba y se restauran al
    // final.
    const size_t saved_flops = cuda::min_matmul_flops();
    const size_t saved_elements = cuda::min_elementwise_elements();
    cuda::set_thresholds(0, 0);

    engine::manual_seed(20260728);

    // --- producto de matrices, las cuatro variantes ---
    //
    // Cada kernel se comprueba por separado, y sobre las mismas formas. Es lo
    // que de verdad protege este trabajo: el kernel de teselas de registros
    // opera sobre bloques de 128x128, así que sus fallos aparecen justo en las
    // formas que no son múltiplo de eso, y sólo ahí. Probar únicamente la
    // variante por defecto dejaría a las otras tres sin red.
    {
        const cuda::MatmulKernel variants[] = {
            cuda::MatmulKernel::Naive,
            cuda::MatmulKernel::Tiled,
            cuda::MatmulKernel::RegisterTiled,
            cuda::MatmulKernel::Vectorized,
        };

        struct Case { size_t M, K, N; };
        const Case cases[] = {
            {1, 1, 1},          // el caso degenerado
            {17, 23, 31},       // por debajo de una sola tesela
            {32, 32, 32},       // justo una tesela de 32
            {33, 65, 129},      // restos en los tres ejes
            {127, 128, 129},    // alrededor del bloque de 128
            {128, 128, 128},    // exactamente un bloque
            {129, 256, 257},    // más de un bloque, con resto
            {256, 260, 256},    // K múltiplo de 4 pero no de 8: tesela K parcial
        };

        for (cuda::MatmulKernel variant : variants) {
            cuda::set_matmul_kernel(variant);
            const std::string tag = std::string("[") + cuda::matmul_kernel_name(variant) + "] ";

            for (const Case& c : cases) {
                Tensor A = Tensor::randn({c.M, c.K});
                Tensor B = Tensor::randn({c.K, c.N});
                compare(tag + "matmul " + std::to_string(c.M) + "x" + std::to_string(c.K) +
                            "x" + std::to_string(c.N),
                        [&] { return A.matmul(B); });
            }

            Tensor QB = Tensor::randn({4, 3, 17, 23});
            Tensor KB = Tensor::randn({4, 3, 23, 11});
            compare(tag + "matmul por lotes (4,3,17,23) x (4,3,23,11)",
                    [&] { return QB.matmul(KB); });

            // Operando 2D compartido con todo el lote: comprueba el paso 0.
            Tensor X = Tensor::randn({8, 12, 40});
            Tensor W = Tensor::randn({40, 20});
            compare(tag + "matmul con operando compartido (8,12,40) x (40,20)",
                    [&] { return X.matmul(W); });
        }

        // Una forma cuya K no es múltiplo de 4: pedir la variante vectorizada
        // tiene que degradarse a la de registros, no leer float4 desalineados.
        // Una lectura desalineada no da error, da otro valor.
        cuda::set_matmul_kernel(cuda::MatmulKernel::Vectorized);
        Tensor A = Tensor::randn({131, 133});
        Tensor B = Tensor::randn({133, 135});
        compare("[vectorized] se degrada con K y N no alineados (133, 135)",
                [&] { return A.matmul(B); });

        cuda::set_matmul_kernel(cuda::MatmulKernel::Auto);
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

    // Un resumen al final: cuantos kernels se ejecutaron de verdad. Si esta
    // linea dice cero, todo lo de arriba se calculo en CPU y no significa nada.
    testing::check(cuda::kernels_failed() == 0,
                   "ningun kernel fallo al lanzarse (" +
                       std::to_string(cuda::kernels_launched()) + " ejecutados, " +
                       std::to_string(cuda::kernels_failed()) + " fallidos)");

    cuda::set_thresholds(saved_flops, saved_elements);
    cuda::set_enabled(true);
}

#else // !ENGINE_CUDA

void run_cuda_parity_tests() {
    testing::section("Paridad CPU / GPU (Fase 6)");
    std::cout << "  (compilado sin CUDA: -DENGINE_CUDA=ON para incluir estas pruebas)\n";
}

#endif // ENGINE_CUDA
