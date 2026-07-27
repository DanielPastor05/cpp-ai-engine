// Banco de pruebas de rendimiento.
//
// No verifica nada: mide. Está en el repositorio para que las cifras de las
// notas de rendimiento del README sean reproducibles y para detectar
// regresiones al tocar los núcleos. No se ejecuta en CI, porque los tiempos de
// un runner compartido no son comparables entre ejecuciones.
//
//   cmake --build build --target bench && ./build/bench

#include "engine/tensor.hpp"
#include "engine/random.hpp"
#include "engine/autograd.hpp"
#include "engine/nn.hpp"
#include "engine/conv.hpp"
#include "engine/transformer.hpp"
#include "engine/optim.hpp"

#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using engine::Tensor;
namespace nn = engine::nn;
namespace optim = engine::optim;

namespace {

// Ejecuta la función hasta acumular al menos min_seconds y devuelve el tiempo
// medio por iteración, para que las operaciones rápidas no queden a merced de
// la resolución del reloj.
double time_op(const std::function<void()>& fn, double min_seconds = 0.3) {
    // Una pasada en frío para no medir el primer fallo de caché
    fn();

    size_t reps = 0;
    const auto start = std::chrono::steady_clock::now();
    double elapsed = 0.0;
    while (elapsed < min_seconds) {
        fn();
        ++reps;
        elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    }
    return elapsed / static_cast<double>(reps);
}

void row(const std::string& label, double seconds, const std::string& note = "") {
    printf("  %-42s %9.3f ms   %s\n", label.c_str(), seconds * 1000.0, note.c_str());
}

void section(const std::string& title) {
    printf("\n=== %s ===\n", title.c_str());
}

std::string gflops(double seconds, double flops) {
    char buf[64];
    snprintf(buf, sizeof(buf), "%.2f GFLOP/s", flops / seconds / 1e9);
    return buf;
}

} // namespace

int main() {
    engine::manual_seed(1);
    printf("Banco de pruebas de cpp-ai-engine\n");

    section("matmul");
    {
        struct Case { size_t M, K, N; };
        const Case cases[] = {{64, 64, 64}, {256, 256, 256}, {512, 128, 512}};
        for (const Case& c : cases) {
            Tensor A = Tensor::randn({c.M, c.K});
            Tensor B = Tensor::randn({c.K, c.N});
            const double t = time_op([&] { Tensor C = A.matmul(B); });
            row(std::to_string(c.M) + "x" + std::to_string(c.K) + "x" + std::to_string(c.N), t,
                gflops(t, 2.0 * c.M * c.K * c.N));
        }
        // Por lotes, como en la atención
        Tensor Q = Tensor::randn({32, 4, 16, 16});
        Tensor K = Tensor::randn({32, 4, 16, 16});
        const double t = time_op([&] { Tensor C = Q.matmul(K); });
        row("por lotes (32, 4, 16, 16)", t, gflops(t, 2.0 * 32 * 4 * 16 * 16 * 16));
    }

    section("operaciones elemento a elemento (1M valores)");
    {
        Tensor A = Tensor::randn({1000, 1000});
        Tensor B = Tensor::randn({1000, 1000});
        Tensor row_vec = Tensor::randn({1000});
        row("suma", time_op([&] { Tensor C = A + B; }));
        row("producto", time_op([&] { Tensor C = A * B; }));
        row("suma con difusion", time_op([&] { Tensor C = A + row_vec; }));
        row("relu", time_op([&] { Tensor C = A.relu(); }));
        row("softmax (por filas)", time_op([&] { Tensor C = A.softmax(); }));
        row("transpose", time_op([&] { Tensor C = A.transpose(); }));
        row("sum(axis=0)", time_op([&] { Tensor C = A.sum(0); }));
    }

    section("permute");
    {
        Tensor T = Tensor::randn({32, 8, 16, 16});
        row("(32,8,16,16) -> {0,2,1,3}", time_op([&] { Tensor C = T.permute({0, 2, 1, 3}); }));
    }

    section("capas (paso adelante + atras)");
    {
        nn::Linear dense(512, 512);
        Tensor x = Tensor::randn({64, 512}, 0.0f, 1.0f, true);
        row("Linear(512,512) lote 64", time_op([&] {
            dense.zero_grad();
            dense(x).sum().backward();
        }));

        nn::Conv2d conv(8, 16, nn::Window2d(3, 3, 1, 1));
        Tensor img = Tensor::randn({16, 8, 16, 16}, 0.0f, 1.0f, true);
        row("Conv2d(8->16, 3x3) lote 16 de 16x16", time_op([&] {
            conv.zero_grad();
            conv(img).sum().backward();
        }));

        nn::MaxPool2d pool(2, 2);
        row("MaxPool2d 2x2", time_op([&] {
            Tensor out = pool(img);
            out.sum().backward();
        }));

        nn::LayerNorm norm(256);
        Tensor seq = Tensor::randn({32, 16, 256}, 0.0f, 1.0f, true);
        row("LayerNorm(256) sobre (32,16,256)", time_op([&] {
            norm.zero_grad();
            norm(seq).sum().backward();
        }));

        nn::TransformerBlock block(128, 4, 256);
        Tensor tokens = Tensor::randn({16, 32, 128}, 0.0f, 1.0f, true);
        row("TransformerBlock(128, 4 cabezas) (16,32,128)", time_op([&] {
            block.zero_grad();
            block(tokens).sum().backward();
        }));
    }

    section("iteracion de entrenamiento completa");
    {
        nn::Sequential mlp{
            nn::make<nn::Linear>(128, 256),
            nn::make<nn::ReLU>(),
            nn::make<nn::Linear>(256, 10)
        };
        optim::Adam opt(mlp.parameters(), 0.001f);
        Tensor X = Tensor::randn({128, 128});
        std::vector<size_t> y(128, 3);

        row("MLP 128-256-10, lote 128", time_op([&] {
            opt.zero_grad();
            nn::cross_entropy_loss(mlp(X), y).backward();
            opt.step();
        }));
    }

    section("inferencia frente a entrenamiento");
    {
        nn::TransformerBlock block(128, 4, 256);
        Tensor tokens = Tensor::randn({16, 32, 128}, 0.0f, 1.0f, true);
        const double with_graph = time_op([&] { Tensor out = block(tokens); });
        const double without = time_op([&] {
            engine::autograd::NoGradGuard no_grad;
            Tensor out = block(tokens);
        });
        row("TransformerBlock construyendo grafo", with_graph);
        row("TransformerBlock bajo NoGradGuard", without,
            "x" + std::to_string(with_graph / without).substr(0, 4) + " mas rapido");
    }

    printf("\n");
    return 0;
}
