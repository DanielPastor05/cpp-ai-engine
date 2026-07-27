// Suite de pruebas del motor: tensores, autograd, capas y optimizadores.
//
// El grueso de la verificación de autograd se hace por comprobación numérica
// de gradientes (diferencias centradas), que es la forma estándar de detectar
// una regla de la cadena mal derivada.

#include "engine/tensor.hpp"
#include "engine/autograd.hpp"
#include "engine/nn.hpp"
#include "engine/optim.hpp"

#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using engine::Tensor;
namespace nn = engine::nn;
namespace optim = engine::optim;

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const std::string& what) {
    ++g_checks;
    if (condition) {
        std::cout << "  [ ok ] " << what << "\n";
    } else {
        ++g_failures;
        std::cout << "  [FAIL] " << what << "\n";
    }
}

void check_close(float actual, float expected, const std::string& what, float tol = 1e-4f) {
    const bool ok = std::fabs(actual - expected) <= tol;
    ++g_checks;
    if (ok) {
        std::cout << "  [ ok ] " << what << "\n";
    } else {
        ++g_failures;
        std::cout << "  [FAIL] " << what << " (obtenido " << actual
                  << ", esperado " << expected << ")\n";
    }
}

void check_throws(const std::function<void()>& fn, const std::string& what) {
    ++g_checks;
    try {
        fn();
        ++g_failures;
        std::cout << "  [FAIL] " << what << " (no lanzo excepcion)\n";
    } catch (const std::exception&) {
        std::cout << "  [ ok ] " << what << "\n";
    }
}

void section(const std::string& title) {
    std::cout << "\n== " << title << " ==\n";
}

// Comprueba el gradiente analítico de `loss_fn` respecto a `input` contra la
// aproximación por diferencias centradas: (f(x+h) - f(x-h)) / 2h.
void check_gradient(const std::string& what, Tensor input,
                    const std::function<Tensor(Tensor&)>& loss_fn,
                    float tol = 2e-2f) {
    input.set_requires_grad(true);
    input.zero_grad();

    Tensor loss = loss_fn(input);
    loss.backward();

    if (!input.has_grad()) {
        ++g_checks;
        ++g_failures;
        std::cout << "  [FAIL] " << what << " (no se propago ningun gradiente)\n";
        return;
    }

    const std::vector<float> analytic = input.grad().data();
    const float h = 1e-3f;
    float max_error = 0.0f;

    for (size_t i = 0; i < input.size(); ++i) {
        engine::autograd::NoGradGuard no_grad; // las evaluaciones numéricas no necesitan grafo
        const float original = input.data()[i];

        input.data()[i] = original + h;
        const float plus = loss_fn(input).data()[0];

        input.data()[i] = original - h;
        const float minus = loss_fn(input).data()[0];

        input.data()[i] = original;

        const float numeric = (plus - minus) / (2.0f * h);
        max_error = std::max(max_error, std::fabs(numeric - analytic[i]));
    }

    ++g_checks;
    if (max_error <= tol) {
        std::cout << "  [ ok ] " << what << " (error maximo " << std::scientific
                  << std::setprecision(2) << max_error << std::defaultfloat << ")\n";
    } else {
        ++g_failures;
        std::cout << "  [FAIL] " << what << " (error maximo " << max_error << " > " << tol << ")\n";
    }
}

// ---------------------------------------------------------
// Pruebas
// ---------------------------------------------------------

void test_tensor_basics() {
    section("Tensor: forma, strides e indexacion");

    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    check(A.shape() == std::vector<size_t>({2, 3}), "la forma es (2, 3)");
    check(A.strides() == std::vector<size_t>({3, 1}), "los strides row-major son (3, 1)");
    check(A.size() == 6, "el tensor tiene 6 elementos");
    check_close(A({1, 2}), 6.0f, "A[1, 2] == 6");

    check_throws([&] { A({2, 0}); }, "indexar fuera de rango lanza excepcion");
    check_throws([&] { Tensor({2, 3}, {1.0f, 2.0f}); }, "datos de tamano incorrecto lanzan excepcion");
    check_throws([&] { A.reshape({4, 2}); }, "reshape incompatible lanza excepcion");

    Tensor R = A.reshape({3, 2});
    check(R.shape() == std::vector<size_t>({3, 2}), "reshape a (3, 2) conserva los datos");
    check_close(R({2, 1}), 6.0f, "reshape mantiene el orden de memoria");
}

void test_matmul() {
    section("Tensor: multiplicacion matricial");

    Tensor M1({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor M2({3, 2}, {7, 8, 9, 1, 2, 3});
    Tensor R = M1.matmul(M2);

    check(R.shape() == std::vector<size_t>({2, 2}), "(2,3) x (3,2) da (2,2)");
    check_close(R({0, 0}), 31.0f, "R[0,0] == 31");
    check_close(R({0, 1}), 19.0f, "R[0,1] == 19");
    check_close(R({1, 0}), 85.0f, "R[1,0] == 85");
    check_close(R({1, 1}), 55.0f, "R[1,1] == 55");

    check_throws([&] { M1.matmul(M1); }, "dimensiones internas incompatibles lanzan excepcion");

    Tensor T = M1.transpose();
    check(T.shape() == std::vector<size_t>({3, 2}), "transpose invierte la forma");
    check_close(T({2, 1}), 6.0f, "transpose intercambia los indices");
}

void test_broadcast_add() {
    section("Tensor: difusion del vector fila en la suma");

    Tensor X({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor b({1, 3}, {10, 20, 30}, true);
    Tensor R = X + b;

    check_close(R({0, 0}), 11.0f, "la primera fila recibe el sesgo");
    check_close(R({1, 2}), 36.0f, "la segunda fila recibe el mismo sesgo");

    Tensor loss = R.sum();
    loss.backward();
    check_close(b.grad().data()[0], 2.0f, "el gradiente del sesgo suma por columnas (2 filas)");

    Tensor bad({1, 4}, 0.0f);
    check_throws([&] { X + bad; }, "una difusion con anchura distinta lanza excepcion");
}

void test_autograd_scalar() {
    section("Autograd: derivadas analiticas conocidas");

    // L = a*b + relu(a), con a=2, b=3 -> dL/da = b+1 = 4, dL/db = a = 2
    Tensor a({1}, {2.0f}, true);
    Tensor b({1}, {3.0f}, true);
    Tensor L = (a * b) + a.relu();
    L.backward();

    check_close(L.data()[0], 8.0f, "L = a*b + relu(a) = 8");
    check_close(a.grad().data()[0], 4.0f, "dL/da == b + 1 == 4");
    check_close(b.grad().data()[0], 2.0f, "dL/db == a == 2");

    // Un nodo reutilizado debe acumular gradiente por ambas ramas: L = x + x
    Tensor x({1}, {5.0f}, true);
    Tensor y = (x + x).sum();
    y.backward();
    check_close(x.grad().data()[0], 2.0f, "un nodo compartido acumula gradiente de ambas ramas");

    // Llamar backward dos veces acumula (igual que PyTorch)
    Tensor z({1}, {3.0f}, true);
    Tensor w = (z * 2.0f).sum();
    w.backward();
    Tensor w2 = (z * 2.0f).sum();
    w2.backward();
    check_close(z.grad().data()[0], 4.0f, "backward acumula gradientes entre llamadas");

    z.zero_grad();
    check_close(z.grad().data()[0], 0.0f, "zero_grad limpia el gradiente");
}

void test_autograd_numeric() {
    section("Autograd: verificacion numerica de gradientes");

    // Ningún valor debe caer exactamente en 0: ReLU no es derivable en ese
    // punto y la diferencia centrada daría 0.5 frente al 0 analítico.
    Tensor A({3, 4}, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A.data()[i] = 0.5f * static_cast<float>(i) - 2.25f;

    check_gradient("gradiente de sum()", A, [](Tensor& t) { return t.sum(); });
    check_gradient("gradiente de mean()", A, [](Tensor& t) { return t.mean(); });
    check_gradient("gradiente de relu()", A, [](Tensor& t) { return t.relu().sum(); });
    check_gradient("gradiente de transpose()", A, [](Tensor& t) { return t.transpose().sum(); });
    check_gradient("gradiente de reshape()", A, [](Tensor& t) { return t.reshape({2, 6}).sum(); });
    check_gradient("gradiente de x*x (Hadamard)", A, [](Tensor& t) { return (t * t).sum(); });
    check_gradient("gradiente de softmax()", A, [](Tensor& t) {
        Tensor w({3, 4}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 1, 2, 3});
        return (t.softmax() * w).sum();
    });
    check_gradient("gradiente de matmul()", A, [](Tensor& t) {
        Tensor B({4, 2}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f});
        return t.matmul(B).sum();
    });
    check_gradient("gradiente de la division", A, [](Tensor& t) {
        Tensor d({3, 4}, 3.0f);
        return (t / d).sum();
    });
    check_gradient("gradiente de una composicion profunda", A, [](Tensor& t) {
        Tensor B({4, 3}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f, 1, 2, -3, 0.75f});
        Tensor h = t.matmul(B).relu();
        return (h * h).mean();
    });

    Tensor logits({4, 3}, {0.4f, -1.2f, 2.0f, 1.1f, 0.3f, -0.7f,
                           -2.0f, 0.9f, 0.2f, 0.6f, 0.6f, -1.5f});
    std::vector<size_t> targets = {2, 0, 1, 1};
    check_gradient("gradiente de cross_entropy_loss()", logits, [&](Tensor& t) {
        return nn::cross_entropy_loss(t, targets);
    });
}

void test_no_grad_and_errors() {
    section("Autograd: NoGradGuard y errores");

    Tensor a({2, 2}, 1.0f, true);
    {
        engine::autograd::NoGradGuard no_grad;
        Tensor b = (a * 2.0f).sum();
        check(!b.requires_grad(), "NoGradGuard evita registrar el grafo");
        check(b.get_impl()->parents.empty(), "el resultado no guarda padres bajo NoGradGuard");
    }
    check(engine::autograd::grad_enabled(), "el modo autograd se restaura al salir del guard");

    Tensor c({2, 2}, 1.0f, true);
    Tensor d = c * 2.0f;
    check_throws([&] { d.backward(); }, "backward implicito sobre un no-escalar lanza excepcion");

    d.backward(Tensor({2, 2}, 1.0f));
    check_close(c.grad().data()[0], 2.0f, "backward(grad_output) explicito si funciona");

    Tensor e({2, 2}, 1.0f, true);
    check_throws([&] { e.add_grad(Tensor({3, 3}, 1.0f)); },
                 "add_grad con forma incompatible lanza excepcion");
    check_throws([&] { Tensor({2, 2}, 1.0f).grad(); },
                 "acceder a un gradiente inexistente lanza excepcion");
}

void test_graph_is_released() {
    section("Autograd: el grafo se libera (sin ciclos de shared_ptr)");

    // Si backward_fn capturase su propio tensor de salida se formaría un ciclo
    // de shared_ptr y el nodo no se destruiría nunca al salir del ámbito.
    std::weak_ptr<engine::TensorImpl> weak_node;
    {
        Tensor a({4, 4}, 2.0f, true);
        Tensor intermediate = (a * a).relu();
        weak_node = intermediate.get_impl();
        Tensor loss = intermediate.sum();
        loss.backward();
        check(!weak_node.expired(), "el nodo intermedio sigue vivo dentro del ambito");
    }
    check(weak_node.expired(), "el nodo intermedio se libera al salir del ambito");
}

void test_nn_layers() {
    section("nn: capas");

    engine::manual_seed(7);

    nn::Linear layer(4, 3);
    check(layer.weight().shape() == std::vector<size_t>({4, 3}), "Linear crea pesos (in, out)");
    check(layer.bias().shape() == std::vector<size_t>({1, 3}), "Linear crea un sesgo (1, out)");
    check(layer.num_parameters() == 15, "Linear(4,3) tiene 4*3 + 3 = 15 parametros");

    Tensor input({5, 4}, 1.0f);
    Tensor out = layer(input);
    check(out.shape() == std::vector<size_t>({5, 3}), "Linear preserva el tamano del lote");

    check_throws([&] { layer(Tensor({5, 7}, 1.0f)); },
                 "una entrada con in_features erroneo lanza excepcion");
    check_throws([&] { layer(Tensor({5}, 1.0f)); }, "una entrada 1D lanza excepcion");

    nn::Linear no_bias(4, 3, false);
    check(no_bias.parameters().size() == 1, "Linear sin sesgo expone un solo parametro");

    nn::Sequential model{
        nn::make<nn::Linear>(4, 8),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(8, 2)
    };
    check(model.parameters().size() == 4, "Sequential agrega los parametros de sus capas");
    check(model.num_parameters() == 4 * 8 + 8 + 8 * 2 + 2, "Sequential suma bien los parametros");
    check(model(input).shape() == std::vector<size_t>({5, 2}), "Sequential encadena las capas");

    check_throws([&] { nn::Sequential s; s.add(nullptr); }, "Sequential rechaza capas nulas");
}

void test_softmax_and_losses() {
    section("nn: softmax y funciones de perdida");

    Tensor logits({2, 3}, {1.0f, 2.0f, 3.0f, 1.0f, 1.0f, 1.0f});
    Tensor probs = logits.softmax();

    float row0 = probs({0, 0}) + probs({0, 1}) + probs({0, 2});
    float row1 = probs({1, 0}) + probs({1, 1}) + probs({1, 2});
    check_close(row0, 1.0f, "cada fila del softmax suma 1");
    check_close(row1, 1.0f, "la fila uniforme tambien suma 1");
    check_close(probs({1, 0}), 1.0f / 3.0f, "logits iguales dan probabilidades iguales");
    check(probs({0, 2}) > probs({0, 1}), "el softmax conserva el orden de los logits");

    // Estabilidad numérica: sin restar el máximo, exp(1000) desbordaría
    Tensor huge({1, 3}, {1000.0f, 1000.0f, 1000.0f});
    Tensor huge_probs = huge.softmax();
    check_close(huge_probs({0, 0}), 1.0f / 3.0f, "el softmax es estable con logits enormes");

    // Entropía cruzada de una prediccion perfecta -> ~0
    Tensor confident({1, 3}, {50.0f, 0.0f, 0.0f});
    check_close(nn::cross_entropy_loss(confident, {0}).data()[0], 0.0f,
                "la entropia cruzada de una prediccion perfecta es ~0");

    // Distribución uniforme sobre C clases -> log(C)
    Tensor uniform({1, 4}, 0.0f);
    check_close(nn::cross_entropy_loss(uniform, {0}).data()[0], std::log(4.0f),
                "la entropia cruzada uniforme vale log(C)");

    check_throws([&] { nn::cross_entropy_loss(logits, {0}); },
                 "un numero de etiquetas incorrecto lanza excepcion");
    check_throws([&] { nn::cross_entropy_loss(logits, {0, 9}); },
                 "una etiqueta fuera de rango lanza excepcion");

    // MSE
    Tensor pred({1, 3}, {1.0f, 2.0f, 3.0f});
    Tensor target({1, 3}, {1.0f, 4.0f, 3.0f});
    check_close(nn::mse_loss(pred, target).data()[0], 4.0f / 3.0f, "mse_loss calcula la media");

    // Métricas
    check(nn::argmax_rows(logits) == std::vector<size_t>({2, 0}), "argmax_rows toma el maximo por fila");
    check_close(nn::accuracy(logits, {2, 0}), 1.0f, "accuracy con todo correcto es 1");
    check_close(nn::accuracy(logits, {0, 0}), 0.5f, "accuracy con la mitad correcta es 0.5");
}

void test_optimizers() {
    section("optim: SGD y Adam");

    // Minimizar f(w) = (w - 3)^2, cuyo mínimo está en w = 3
    auto minimize = [](optim::Optimizer& opt, Tensor& w, int steps) {
        for (int i = 0; i < steps; ++i) {
            opt.zero_grad();
            Tensor diff = w - Tensor({1}, {3.0f});
            Tensor loss = (diff * diff).sum();
            loss.backward();
            opt.step();
        }
    };

    Tensor w_sgd({1}, {0.0f}, true);
    optim::SGD sgd({w_sgd}, 0.1f);
    minimize(sgd, w_sgd, 200);
    check_close(w_sgd.data()[0], 3.0f, "SGD converge al minimo", 1e-3f);

    Tensor w_mom({1}, {0.0f}, true);
    optim::SGD sgd_mom({w_mom}, 0.05f, 0.9f);
    minimize(sgd_mom, w_mom, 200);
    check_close(w_mom.data()[0], 3.0f, "SGD con momento converge al minimo", 1e-3f);

    Tensor w_adam({1}, {0.0f}, true);
    optim::Adam adam({w_adam}, 0.1f);
    minimize(adam, w_adam, 300);
    check_close(w_adam.data()[0], 3.0f, "Adam converge al minimo", 1e-3f);

    // La corrección de sesgo hace que el primer paso de Adam valga ~lr
    Tensor w_first({1}, {0.0f}, true);
    optim::Adam adam_first({w_first}, 0.1f);
    adam_first.zero_grad();
    Tensor l = (w_first * 2.0f).sum();
    l.backward();
    adam_first.step();
    check_close(w_first.data()[0], -0.1f,
                "el primer paso de Adam vale ~lr gracias a la correccion de sesgo", 1e-3f);
    check(adam_first.steps() == 1, "Adam cuenta los pasos aplicados");

    // Un parámetro sin gradiente no debe moverse
    Tensor untouched({2}, {1.0f, 2.0f}, true);
    optim::SGD idle({untouched}, 0.5f);
    idle.step();
    check_close(untouched.data()[0], 1.0f, "un parametro sin gradiente no se modifica");

    check_throws([&] { optim::SGD({w_sgd}, -1.0f); }, "un learning rate negativo lanza excepcion");
    check_throws([&] { optim::Adam({w_sgd}, 0.1f, 1.5f); }, "un beta1 invalido lanza excepcion");
}

void test_end_to_end_training() {
    section("Entrenamiento de extremo a extremo");

    engine::manual_seed(123);

    // XOR: el caso mínimo que no es separable linealmente
    Tensor X({4, 2}, {0, 0,
                      0, 1,
                      1, 0,
                      1, 1}, false);
    std::vector<size_t> y = {0, 1, 1, 0};

    nn::Sequential model{
        nn::make<nn::Linear>(2, 16),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(16, 2)
    };
    optim::Adam opt(model.parameters(), 0.1f);

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 400; ++epoch) {
        opt.zero_grad();
        Tensor logits = model(X);
        Tensor loss = nn::cross_entropy_loss(logits, y);
        loss.backward();
        opt.step();

        if (epoch == 0) first_loss = loss.data()[0];
        last_loss = loss.data()[0];
    }

    check(last_loss < first_loss, "la perdida disminuye durante el entrenamiento");
    check(last_loss < 0.05f, "el MLP ajusta el XOR (perdida < 0.05)");
    check_close(nn::accuracy(model(X), y), 1.0f, "el MLP clasifica el XOR con 100% de exactitud");

    // El mismo problema con un modelo lineal no puede resolverse
    engine::manual_seed(123);
    nn::Linear linear(2, 2);
    optim::Adam linear_opt(linear.parameters(), 0.1f);
    for (int epoch = 0; epoch < 400; ++epoch) {
        linear_opt.zero_grad();
        Tensor loss = nn::cross_entropy_loss(linear(X), y);
        loss.backward();
        linear_opt.step();
    }
    check(nn::accuracy(linear(X), y) < 1.0f, "un modelo lineal no resuelve el XOR (control)");
}

} // namespace

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Suite de pruebas de cpp-ai-engine                 \n";
    std::cout << "====================================================\n";

    test_tensor_basics();
    test_matmul();
    test_broadcast_add();
    test_autograd_scalar();
    test_autograd_numeric();
    test_no_grad_and_errors();
    test_graph_is_released();
    test_nn_layers();
    test_softmax_and_losses();
    test_optimizers();
    test_end_to_end_training();

    std::cout << "\n====================================================\n";
    if (g_failures == 0) {
        std::cout << "  TODAS LAS PRUEBAS PASARON (" << g_checks << " comprobaciones)\n";
    } else {
        std::cout << "  " << g_failures << " DE " << g_checks << " COMPROBACIONES FALLARON\n";
    }
    std::cout << "====================================================\n";

    return g_failures == 0 ? 0 : 1;
}
