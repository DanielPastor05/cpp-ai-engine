// Suite de pruebas del motor: tensores, autograd, capas y optimizadores.
//
// El grueso de la verificación de autograd se hace por comprobación numérica
// de gradientes (diferencias centradas), que es la forma estándar de detectar
// una regla de la cadena mal derivada.

#include "engine/tensor.hpp"
#include "engine/autograd.hpp"
#include "engine/nn.hpp"
#include "engine/conv.hpp"
#include "engine/transformer.hpp"
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

std::string win_name(const nn::Window2d& w) {
    return "k=" + std::to_string(w.kernel_h) + "x" + std::to_string(w.kernel_w) +
           ", s=" + std::to_string(w.stride) + ", p=" + std::to_string(w.padding);
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

    // Regresión: con más ejes que la base pero unos iniciales, el número de
    // repeticiones se calculaba con un producto sobre un rango vacío que se
    // confundía con el producto completo, y el backward leía fuera del búfer.
    Tensor same_rank({3, 4}, 1.0f, false);
    Tensor leading_one({1, 3, 4}, 2.0f, true);
    Tensor bc = same_rank + leading_one;
    check(bc.shape() == std::vector<size_t>({3, 4}), "difundir (1,3,4) sobre (3,4) conserva la forma");
    check_close(bc.data()[0], 3.0f, "difundir con un eje inicial de tamano 1 suma bien");
    bc.sum().backward();
    check_close(leading_one.grad().data()[0], 1.0f,
                "su gradiente es 1, no la suma de repeticiones inexistentes");

    // Un escalar tambien se difunde sobre cualquier forma
    Tensor scalar({1}, std::vector<float>{5.0f}, true);
    Tensor plus_scalar = X + scalar;
    check_close(plus_scalar.data()[0], 6.0f, "un tensor de un elemento se difunde como escalar");
    plus_scalar.sum().backward();
    check_close(scalar.grad().data()[0], 6.0f, "el escalar acumula el gradiente de los 6 elementos");
}

void test_autograd_scalar() {
    section("Autograd: derivadas analiticas conocidas");

    // L = a*b + relu(a), con a=2, b=3 -> dL/da = b+1 = 4, dL/db = a = 2
    Tensor a({1}, std::vector<float>{2.0f}, true);
    Tensor b({1}, std::vector<float>{3.0f}, true);
    Tensor L = (a * b) + a.relu();
    L.backward();

    check_close(L.data()[0], 8.0f, "L = a*b + relu(a) = 8");
    check_close(a.grad().data()[0], 4.0f, "dL/da == b + 1 == 4");
    check_close(b.grad().data()[0], 2.0f, "dL/db == a == 2");

    // Un nodo reutilizado debe acumular gradiente por ambas ramas: L = x + x
    Tensor x({1}, std::vector<float>{5.0f}, true);
    Tensor y = (x + x).sum();
    y.backward();
    check_close(x.grad().data()[0], 2.0f, "un nodo compartido acumula gradiente de ambas ramas");

    // Llamar backward dos veces acumula (igual que PyTorch)
    Tensor z({1}, std::vector<float>{3.0f}, true);
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

void test_repeated_backward() {
    section("Autograd: backward repetido sobre el mismo grafo");

    // Regresión: los nodos intermedios conservaban el gradiente de la llamada
    // anterior, así que el segundo recorrido propagaba la suma de ambos y
    // multiplicaba el gradiente de las hojas (daba 18 en lugar de 6).
    Tensor x({1}, std::vector<float>{1.0f}, true);
    Tensor h = x * 2.0f;
    Tensor L = (h * 3.0f).sum();

    L.backward();
    check_close(x.grad().data()[0], 6.0f, "primer backward: dL/dx == 6");

    x.zero_grad();
    L.backward();
    check_close(x.grad().data()[0], 6.0f, "segundo backward sobre el mismo grafo: dL/dx == 6");

    // Sin limpiar la hoja, el resultado debe ser exactamente el doble:
    // solo las hojas acumulan.
    L.backward();
    check_close(x.grad().data()[0], 12.0f, "sin zero_grad las hojas acumulan (6 + 6)");
}

void test_row_indexing_and_batches() {
    section("Tensor: indexacion (fila, col) y mini-lotes");

    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    check_close(A(1, 2), 6.0f, "A(1, 2) accede sin reservar memoria dinamica");
    check_close(A(0, 0), 1.0f, "A(0, 0) es el primer elemento");
    A(0, 1) = 99.0f;
    check_close(A({0, 1}), 99.0f, "A(fila, col) permite escritura");
    check_throws([&] { A(2, 0); }, "A(fila, col) fuera de rango lanza excepcion");
    check_throws([&] { Tensor({4}, 1.0f)(0, 0); }, "A(fila, col) sobre un tensor 1D lanza excepcion");

    // Escalar a la izquierda
    Tensor t({2}, {1.0f, 2.0f});
    check_close((2.0f * t).data()[1], 4.0f, "2.0f * t multiplica por la izquierda");
    check_close((1.0f + t).data()[0], 2.0f, "1.0f + t suma por la izquierda");
    check_close((10.0f - t).data()[1], 8.0f, "10.0f - t resta por la izquierda");

    // select_rows
    Tensor X({4, 2}, {0, 0,
                      1, 1,
                      2, 2,
                      3, 3}, true);
    Tensor batch = X.select_rows({3, 1});
    check(batch.shape() == std::vector<size_t>({2, 2}), "select_rows da (n_indices, cols)");
    check_close(batch(0, 0), 3.0f, "select_rows respeta el orden pedido");
    check_close(batch(1, 0), 1.0f, "select_rows toma la fila correcta");

    // Con índices repetidos el gradiente debe acumularse en la fila de origen
    X.zero_grad();
    Tensor loss = X.select_rows({0, 0, 2}).sum();
    loss.backward();
    check_close(X.grad()(0, 0), 2.0f, "un indice repetido acumula gradiente");
    check_close(X.grad()(1, 0), 0.0f, "una fila no seleccionada no recibe gradiente");
    check_close(X.grad()(2, 0), 1.0f, "una fila seleccionada una vez recibe gradiente 1");

    check_throws([&] { X.select_rows({}); }, "select_rows sin indices lanza excepcion");
    check_throws([&] { X.select_rows({9}); }, "select_rows con un indice invalido lanza excepcion");

    Tensor G({3, 4}, 0.0f);
    for (size_t i = 0; i < G.size(); ++i) G.data()[i] = 0.5f * static_cast<float>(i) - 2.25f;
    check_gradient("gradiente de select_rows()", G, [](Tensor& t) {
        return t.select_rows({2, 0, 2}).sum();
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
            Tensor diff = w - Tensor({1}, std::vector<float>{3.0f});
            Tensor loss = (diff * diff).sum();
            loss.backward();
            opt.step();
        }
    };

    Tensor w_sgd({1}, std::vector<float>{0.0f}, true);
    optim::SGD sgd({w_sgd}, 0.1f);
    minimize(sgd, w_sgd, 200);
    check_close(w_sgd.data()[0], 3.0f, "SGD converge al minimo", 1e-3f);

    Tensor w_mom({1}, std::vector<float>{0.0f}, true);
    optim::SGD sgd_mom({w_mom}, 0.05f, 0.9f);
    minimize(sgd_mom, w_mom, 200);
    check_close(w_mom.data()[0], 3.0f, "SGD con momento converge al minimo", 1e-3f);

    Tensor w_adam({1}, std::vector<float>{0.0f}, true);
    optim::Adam adam({w_adam}, 0.1f);
    minimize(adam, w_adam, 300);
    check_close(w_adam.data()[0], 3.0f, "Adam converge al minimo", 1e-3f);

    // La corrección de sesgo hace que el primer paso de Adam valga ~lr
    Tensor w_first({1}, std::vector<float>{0.0f}, true);
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

void test_im2col() {
    section("conv: im2col y col2im");

    // Ventana 3x3 sobre una imagen 4x4 -> 2x2 posiciones, filas de 9 valores
    Tensor img({1, 1, 4, 4}, { 1,  2,  3,  4,
                               5,  6,  7,  8,
                               9, 10, 11, 12,
                              13, 14, 15, 16}, false);
    nn::Window2d w3(3, 3, 1, 0);

    check(w3.out_h(4) == 2 && w3.out_w(4) == 2, "(4 - 3)/1 + 1 == 2 posiciones por eje");
    check(nn::Window2d(3, 3, 1, 1).out_h(4) == 4, "el relleno 1 con kernel 3 preserva el tamano");
    check(nn::Window2d(2, 2, 2, 0).out_h(4) == 2, "un paso de 2 con kernel 2 divide el tamano");

    Tensor cols = nn::im2col(img, w3);
    check(cols.shape() == std::vector<size_t>({4, 9}), "im2col da (N*oH*oW, C*kH*kW)");
    // Primera ventana: esquina superior izquierda 3x3
    check_close(cols(0, 0), 1.0f, "la primera ventana empieza en el pixel (0,0)");
    check_close(cols(0, 8), 11.0f, "la primera ventana termina en el pixel (2,2)");
    // Última ventana: desplazada un pixel en ambos ejes
    check_close(cols(3, 0), 6.0f, "la ultima ventana empieza en el pixel (1,1)");
    check_close(cols(3, 8), 16.0f, "la ultima ventana termina en el pixel (3,3)");

    // Con relleno, las esquinas del kernel caen fuera y valen cero
    Tensor padded_cols = nn::im2col(img, nn::Window2d(3, 3, 1, 1));
    check(padded_cols.shape() == std::vector<size_t>({16, 9}), "con relleno hay 16 ventanas");
    check_close(padded_cols(0, 0), 0.0f, "la zona de relleno aporta ceros");
    check_close(padded_cols(0, 4), 1.0f, "el centro de la primera ventana rellenada es el pixel (0,0)");

    // col2im devuelve la forma original y acumula los solapes
    Tensor ones({4, 9}, 1.0f, false);
    Tensor scattered = nn::col2im(ones, {1, 1, 4, 4}, w3);
    check(scattered.shape() == std::vector<size_t>({1, 1, 4, 4}), "col2im restaura la forma de entrada");
    check_close(scattered.data()[0], 1.0f, "una esquina pertenece a una sola ventana");
    check_close(scattered.data()[5], 4.0f, "el pixel (1,1) pertenece a las cuatro ventanas");

    // Prueba del adjunto: <im2col(x), y> == <x, col2im(y)>.
    // Es la afirmación exacta de que col2im es la traspuesta de im2col, y por
    // tanto la derivada correcta de la convolución.
    engine::manual_seed(31);
    for (const nn::Window2d& win : {nn::Window2d(3, 3, 1, 0),
                                    nn::Window2d(3, 3, 1, 1),
                                    nn::Window2d(2, 2, 2, 0),
                                    nn::Window2d(2, 3, 1, 1)}) {
        Tensor x = Tensor::randn({2, 3, 5, 6});
        Tensor cols_x = nn::im2col(x, win);
        Tensor y = Tensor::randn(cols_x.shape());
        Tensor back = nn::col2im(y, x.shape(), win);

        float lhs = 0.0f;
        for (size_t i = 0; i < cols_x.size(); ++i) lhs += cols_x.data()[i] * y.data()[i];
        float rhs = 0.0f;
        for (size_t i = 0; i < x.size(); ++i) rhs += x.data()[i] * back.data()[i];

        check_close(lhs, rhs, "col2im es el adjunto de im2col para " + win_name(win), 1e-2f);
    }

    check_throws([&] { nn::im2col(Tensor({4, 4}, 1.0f), w3); }, "im2col sobre un tensor 2D lanza excepcion");
    check_throws([&] { nn::im2col(img, nn::Window2d(5, 5, 1, 0)); },
                 "un kernel mayor que la imagen lanza excepcion");
    check_throws([&] { nn::col2im(ones, {1, 1, 8, 8}, w3); },
                 "col2im con columnas de tamano incoherente lanza excepcion");
}

void test_conv_layers() {
    section("conv: Conv2d, MaxPool2d y Flatten");

    engine::manual_seed(11);

    // Forma de salida
    nn::Conv2d conv(3, 5, nn::Window2d(3, 3, 1, 1));
    Tensor input = Tensor::randn({2, 3, 8, 8});
    Tensor out = conv(input);
    check(out.shape() == std::vector<size_t>({2, 5, 8, 8}), "Conv2d con relleno 1 preserva el tamano");
    check(conv.weight().shape() == std::vector<size_t>({5, 3, 3, 3}),
          "los pesos son (outC, inC, kH, kW)");
    check(conv.num_parameters() == 5 * 3 * 3 * 3 + 5, "Conv2d cuenta pesos y sesgos");

    nn::Conv2d strided(3, 2, nn::Window2d(3, 3, 2, 0));
    check(strided(input).shape() == std::vector<size_t>({2, 2, 3, 3}),
          "Conv2d con paso 2 y sin relleno reduce 8 -> 3");

    check_throws([&] { conv(Tensor({2, 8, 8}, 1.0f)); }, "Conv2d con una entrada 3D lanza excepcion");
    check_throws([&] { conv(Tensor::randn({2, 4, 8, 8})); },
                 "Conv2d con un numero de canales erroneo lanza excepcion");
    check_throws([&] { nn::Conv2d(0, 4, nn::Window2d(3)); }, "Conv2d sin canales de entrada lanza excepcion");

    // Convolución 1x1 con un peso conocido: la salida es un reescalado exacto
    nn::Conv2d unit(1, 1, nn::Window2d(1, 1, 1, 0));
    unit.weight() = Tensor({1, 1, 1, 1}, std::vector<float>{2.0f}, true);
    unit.bias() = Tensor({1}, std::vector<float>{0.5f}, true);
    Tensor pixels({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, false);
    Tensor scaled = unit(pixels);
    check_close(scaled.data()[0], 2.5f, "conv 1x1: 1*2 + 0.5 == 2.5");
    check_close(scaled.data()[3], 8.5f, "conv 1x1: 4*2 + 0.5 == 8.5");

    // Kernel 3x3 de unos sin relleno: cada salida es la suma del bloque 3x3
    nn::Conv2d summer(1, 1, nn::Window2d(3, 3, 1, 0), false);
    summer.weight() = Tensor({1, 1, 3, 3}, std::vector<float>(9, 1.0f), true);
    Tensor grid({1, 1, 3, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9}, false);
    check_close(summer(grid).data()[0], 45.0f, "un kernel de unos suma el bloque completo");

    // MaxPool2d
    nn::MaxPool2d pool(2, 2);
    Tensor pool_in({1, 1, 4, 4}, { 1,  2,  9,  4,
                                   5,  6,  7,  8,
                                  13, 10, 11, 12,
                                   3, 14, 15, 16}, false);
    Tensor pooled = pool(pool_in);
    check(pooled.shape() == std::vector<size_t>({1, 1, 2, 2}), "MaxPool 2x2 con paso 2 divide el tamano");
    check_close(pooled.data()[0], 6.0f, "maximo del bloque superior izquierdo");
    check_close(pooled.data()[1], 9.0f, "maximo del bloque superior derecho");
    check_close(pooled.data()[2], 14.0f, "maximo del bloque inferior izquierdo");
    check_close(pooled.data()[3], 16.0f, "maximo del bloque inferior derecho");
    check_throws([&] { pool(Tensor({4, 4}, 1.0f)); }, "MaxPool2d con una entrada 2D lanza excepcion");
    // Con relleno >= kernel habria ventanas sin ningun valor real que maximizar
    check_throws([&] { nn::MaxPool2d(nn::Window2d(2, 2, 1, 2)); },
                 "MaxPool2d con relleno mayor o igual que el kernel lanza excepcion");

    // El gradiente va solo a la posición ganadora
    Tensor pool_grad({1, 1, 2, 2}, {1.0f, 2.0f, 3.0f, 4.0f}, true);
    nn::MaxPool2d pool2(2, 2);
    Tensor small_out = pool2(pool_grad);
    small_out.sum().backward();
    check_close(pool_grad.grad().data()[3], 1.0f, "el maximo recibe todo el gradiente");
    check_close(pool_grad.grad().data()[0], 0.0f, "los perdedores no reciben gradiente");

    // Flatten
    nn::Flatten flat;
    check(flat(Tensor({2, 3, 4, 5}, 1.0f)).shape() == std::vector<size_t>({2, 60}),
          "Flatten conserva el lote y aplana el resto");
    check_throws([&] { flat(Tensor({6}, 1.0f)); }, "Flatten con una entrada 1D lanza excepcion");
}

void test_conv_gradients() {
    section("conv: verificacion numerica de gradientes");

    engine::manual_seed(5);

    // Gradiente respecto a la entrada, con y sin relleno y con paso 2
    {
        nn::Conv2d conv(2, 3, nn::Window2d(3, 3, 1, 1));
        Tensor x = Tensor::randn({2, 2, 5, 5});
        Tensor w = Tensor::randn({2, 3, 5, 5});
        check_gradient("gradiente de Conv2d respecto a la entrada", x,
                       [&](Tensor& t) { return (conv(t) * w).sum(); });
    }
    {
        nn::Conv2d conv(1, 2, nn::Window2d(2, 2, 2, 0));
        Tensor x = Tensor::randn({2, 1, 6, 6});
        check_gradient("gradiente de Conv2d con paso 2 respecto a la entrada", x,
                       [&](Tensor& t) { return conv(t).sum(); });
    }

    // Gradiente respecto a los pesos y al sesgo. El tensor perturbado es el
    // propio parámetro de la capa (comparten implementación), así que la capa
    // ve el cambio y basta con reevaluar el forward.
    {
        nn::Conv2d conv(2, 3, nn::Window2d(3, 3, 1, 1));
        Tensor x = Tensor::randn({2, 2, 4, 4});
        // Ponderacion no uniforme: con sum() a secas el gradiente de salida es
        // todo unos, y eso puede ocultar un error de indexacion.
        Tensor w = Tensor::randn({2, 3, 4, 4});

        check_gradient("gradiente de Conv2d respecto a los pesos", conv.weight(),
                       [&](Tensor&) { return (conv(x) * w).sum(); });
        check_gradient("gradiente de Conv2d respecto al sesgo", conv.bias(),
                       [&](Tensor&) { return (conv(x) * w).sum(); });
    }

    // MaxPool: valores bien separados para que ninguna perturbacion cambie el
    // ganador de una ventana (el maximo no es derivable en un empate).
    {
        Tensor x({1, 2, 4, 4}, 0.0f);
        for (size_t i = 0; i < x.size(); ++i) {
            x.data()[i] = static_cast<float>((i * 37) % 64) * 0.25f;
        }
        nn::MaxPool2d pool(2, 2);
        check_gradient("gradiente de MaxPool2d", x, [&](Tensor& t) { return pool(t).sum(); });
    }

    // Flatten y una pila completa conv -> relu -> pool -> flatten -> linear
    {
        Tensor x = Tensor::randn({2, 3, 4, 4});
        nn::Flatten flat;
        check_gradient("gradiente de Flatten", x, [&](Tensor& t) { return flat(t).sum(); });
    }
    {
        nn::Sequential net{
            nn::make<nn::Conv2d>(1, 2, nn::Window2d(3, 3, 1, 1)),
            nn::make<nn::ReLU>(),
            nn::make<nn::MaxPool2d>(2, 2),
            nn::make<nn::Flatten>(),
            nn::make<nn::Linear>(8, 2)
        };
        Tensor x = Tensor::randn({2, 1, 4, 4});
        std::vector<size_t> targets = {0, 1};
        check_gradient("gradiente de una CNN completa con entropia cruzada", x,
                       [&](Tensor& t) { return nn::cross_entropy_loss(net(t), targets); });
    }
}

void test_cnn_training() {
    section("conv: entrenamiento de una CNN");

    engine::manual_seed(3);

    // Dos clases separables por una caracteristica local: un pixel brillante
    // arriba a la izquierda o abajo a la derecha, en posiciones variables.
    const size_t N = 40;
    Tensor X({N, 1, 6, 6}, 0.0f, false);
    std::vector<size_t> y(N, 0);
    std::uniform_int_distribution<size_t> jitter(0, 1);

    for (size_t n = 0; n < N; ++n) {
        const size_t label = n % 2;
        const size_t r = (label == 0 ? 0 : 4) + jitter(engine::global_rng());
        const size_t c = (label == 0 ? 0 : 4) + jitter(engine::global_rng());
        X.data()[n * 36 + r * 6 + c] = 1.0f;
        y[n] = label;
    }

    nn::Sequential cnn{
        nn::make<nn::Conv2d>(1, 4, nn::Window2d(3, 3, 1, 1)),
        nn::make<nn::ReLU>(),
        nn::make<nn::MaxPool2d>(2, 2),
        nn::make<nn::Flatten>(),
        nn::make<nn::Linear>(4 * 3 * 3, 2)
    };
    optim::Adam opt(cnn.parameters(), 0.05f);

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 60; ++epoch) {
        opt.zero_grad();
        Tensor loss = nn::cross_entropy_loss(cnn(X), y);
        loss.backward();
        opt.step();
        if (epoch == 0) first_loss = loss.data()[0];
        last_loss = loss.data()[0];
    }

    check(last_loss < first_loss, "la perdida de la CNN disminuye");
    check_close(nn::accuracy(cnn(X), y), 1.0f, "la CNN aprende la tarea al 100%");

    // Los mini-lotes deben funcionar también con volumenes 4D
    Tensor batch = X.select_rows({0, 1, 2});
    check(batch.shape() == std::vector<size_t>({3, 1, 6, 6}),
          "select_rows toma imagenes completas de un tensor 4D");
    check_close(batch.data()[0], X.data()[0], "el mini-lote 4D copia los datos correctos");
}

void test_nd_tensor_ops() {
    section("Tensor: operaciones N-dimensionales");

    // transpose intercambia los dos ultimos ejes en cualquier rango
    Tensor T3({2, 2, 3}, {1, 2, 3, 4, 5, 6,
                          7, 8, 9, 10, 11, 12});
    Tensor Tt = T3.transpose();
    check(Tt.shape() == std::vector<size_t>({2, 3, 2}), "transpose 3D intercambia los dos ultimos ejes");
    check_close(Tt.data()[0], 1.0f, "transpose 3D: primer elemento del primer lote");
    check_close(Tt.data()[1], 4.0f, "transpose 3D: transpone cada matriz del lote");
    check_close(Tt.data()[6], 7.0f, "transpose 3D: el segundo lote es independiente");

    // permute
    Tensor P({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < P.size(); ++i) P.data()[i] = static_cast<float>(i);
    Tensor Pp = P.permute({2, 0, 1});
    check(Pp.shape() == std::vector<size_t>({4, 2, 3}), "permute reordena la forma");
    // El elemento (i, j, k) de P debe estar en (k, i, j) de Pp
    check_close(Pp.data()[(2 * 2 + 1) * 3 + 0], P.data()[(1 * 3 + 0) * 4 + 2],
                "permute reubica los elementos correctamente");
    check(P.permute({0, 1, 2}).shape() == P.shape(), "la permutacion identidad no cambia nada");
    check_throws([&] { P.permute({0, 1}); }, "permute con menos ejes lanza excepcion");
    check_throws([&] { P.permute({0, 1, 1}); }, "permute con un eje repetido lanza excepcion");
    check_throws([&] { P.permute({0, 1, 5}); }, "permute con un eje inexistente lanza excepcion");

    // matmul por lotes: cada matriz del lote se multiplica por separado
    Tensor A({2, 2, 3}, {1, 2, 3, 4, 5, 6,
                         7, 8, 9, 10, 11, 12});
    Tensor B({2, 3, 2}, {1, 0, 0, 1, 1, 1,
                         2, 0, 0, 2, 1, 1});
    Tensor C = A.matmul(B);
    check(C.shape() == std::vector<size_t>({2, 2, 2}), "matmul por lotes da (B, M, N)");
    check_close(C.data()[0], 4.0f, "lote 0: 1*1 + 2*0 + 3*1 == 4");
    check_close(C.data()[1], 5.0f, "lote 0: 1*0 + 2*1 + 3*1 == 5");
    check_close(C.data()[4], 23.0f, "lote 1: 7*2 + 8*0 + 9*1 == 23");

    // Un operando 2D se comparte con todo el lote
    Tensor shared({3, 2}, {1, 0, 0, 1, 1, 1});
    Tensor Cs = A.matmul(shared);
    check(Cs.shape() == std::vector<size_t>({2, 2, 2}), "matmul con matriz 2D compartida da (B, M, N)");
    check_close(Cs.data()[0], 4.0f, "la matriz compartida se aplica al primer lote");
    check_close(Cs.data()[4], 16.0f, "y la misma matriz al segundo (7 + 9)");

    check_throws([&] { Tensor({2, 2, 3}, 1.0f).matmul(Tensor({3, 3, 2}, 1.0f)); },
                 "matmul con lotes distintos lanza excepcion");
    check_throws([&] { Tensor({4}, 1.0f).matmul(Tensor({4}, 1.0f)); },
                 "matmul de vectores 1D lanza excepcion");

    // softmax sobre el ultimo eje de un tensor 3D
    Tensor S3 = Tensor::randn({2, 3, 4}).softmax();
    check(S3.shape() == std::vector<size_t>({2, 3, 4}), "softmax 3D conserva la forma");
    bool all_rows_sum_one = true;
    for (size_t r = 0; r < 6; ++r) {
        float total = 0.0f;
        for (size_t j = 0; j < 4; ++j) total += S3.data()[r * 4 + j];
        if (std::fabs(total - 1.0f) > 1e-4f) all_rows_sum_one = false;
    }
    check(all_rows_sum_one, "cada vector del ultimo eje suma 1 tras el softmax");

    // Difusion por sufijo
    Tensor base({2, 3, 4}, 1.0f);
    Tensor row({3, 4}, 2.0f, true);
    Tensor sum_bc = base + row;
    check(sum_bc.shape() == std::vector<size_t>({2, 3, 4}), "la difusion (3,4) sobre (2,3,4) funciona");
    check_close(sum_bc.data()[0], 3.0f, "la difusion suma el bloque repetido");
    sum_bc.sum().backward();
    check_close(row.grad().data()[0], 2.0f, "el gradiente del operando difundido suma las 2 repeticiones");

    check_throws([&] { base + Tensor({5, 4}, 1.0f); },
                 "una difusion con un sufijo incompatible lanza excepcion");

    // Gradientes numericos de las operaciones nuevas
    Tensor G({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < G.size(); ++i) G.data()[i] = 0.3f * static_cast<float>(i) - 3.1f;

    check_gradient("gradiente de permute()", G, [](Tensor& t) {
        Tensor w = Tensor({4, 2, 3}, 0.0f);
        for (size_t i = 0; i < w.size(); ++i) w.data()[i] = 0.1f * static_cast<float>(i) - 1.0f;
        return (t.permute({2, 0, 1}) * w).sum();
    });
    check_gradient("gradiente de transpose() 3D", G, [](Tensor& t) { return t.transpose().sum(); });
    // El tensor de ponderación se crea FUERA del closure: si se generase
    // dentro, cada evaluación numérica usaría pesos distintos y la
    // comprobación no compararía la misma función consigo misma.
    Tensor w_soft = Tensor::randn({2, 3, 4});
    check_gradient("gradiente de softmax() 3D", G, [&](Tensor& t) {
        return (t.softmax() * w_soft).sum();
    });
    check_gradient("gradiente de matmul por lotes", G, [](Tensor& t) {
        Tensor B2({2, 4, 2}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f,
                              2, 1, -1, 0.5f, 3, -2, 1, 1});
        return t.matmul(B2).sum();
    });
    check_gradient("gradiente de matmul con matriz compartida", G, [](Tensor& t) {
        Tensor shared2({4, 2}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f});
        return t.matmul(shared2).sum();
    });
    {
        // La matriz compartida recibe la suma de las contribuciones del lote
        Tensor shared3({4, 2}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f});
        Tensor batched = Tensor::randn({3, 2, 4});
        check_gradient("gradiente de la matriz compartida en el matmul", shared3,
                       [&](Tensor& t) { return batched.matmul(t).sum(); });
    }

    // Linear sobre entradas de mas de 2 ejes
    engine::manual_seed(19);
    nn::Linear proj(4, 5);
    Tensor seq_input = Tensor::randn({2, 3, 4});
    check(proj(seq_input).shape() == std::vector<size_t>({2, 3, 5}),
          "Linear aplica la proyeccion al ultimo eje de un tensor 3D");
    Tensor w_proj = Tensor::randn({2, 3, 5});
    check_gradient("gradiente de Linear sobre una entrada 3D", seq_input, [&](Tensor& t) {
        return (proj(t) * w_proj).sum();
    });
}

void test_layernorm_and_embedding() {
    section("transformer: LayerNorm y Embedding");

    engine::manual_seed(23);

    nn::LayerNorm norm(4);
    check(norm.num_parameters() == 8, "LayerNorm tiene gamma y beta (4 + 4)");

    Tensor x({2, 4}, {1.0f, 2.0f, 3.0f, 4.0f,
                      10.0f, 10.0f, 10.0f, 10.0f}, false);
    Tensor normed = norm(x);
    check(normed.shape() == x.shape(), "LayerNorm conserva la forma");

    // Primera fila: media 0 y varianza 1 tras normalizar
    float mean = 0.0f;
    for (size_t j = 0; j < 4; ++j) mean += normed.data()[j];
    mean /= 4.0f;
    float var = 0.0f;
    for (size_t j = 0; j < 4; ++j) {
        const float d = normed.data()[j] - mean;
        var += d * d;
    }
    var /= 4.0f;
    check_close(mean, 0.0f, "LayerNorm deja media 0", 1e-3f);
    check_close(var, 1.0f, "LayerNorm deja varianza 1", 1e-2f);

    // Fila constante: sin varianza, epsilon evita dividir por cero
    check_close(normed.data()[4], 0.0f, "una fila constante se normaliza a 0 sin dividir por cero");

    // gamma y beta reescalan
    nn::LayerNorm scaled(4);
    scaled.gamma() = Tensor({4}, 2.0f, true);
    scaled.beta() = Tensor({4}, 5.0f, true);
    Tensor out2 = scaled(x);
    check_close(out2.data()[4], 5.0f, "beta desplaza la salida");

    check_throws([&] { norm(Tensor({2, 3}, 1.0f)); },
                 "LayerNorm con un ultimo eje de otro tamano lanza excepcion");
    check_throws([&] { nn::LayerNorm(0); }, "LayerNorm de tamano cero lanza excepcion");

    Tensor gx = Tensor::randn({3, 4});
    Tensor w_norm = Tensor::randn({3, 4});
    check_gradient("gradiente de LayerNorm respecto a la entrada", gx, [&](Tensor& t) {
        return (norm(t) * w_norm).sum();
    });
    {
        nn::LayerNorm n2(4);
        Tensor fixed = Tensor::randn({3, 4});
        Tensor w = Tensor::randn({3, 4});
        check_gradient("gradiente de LayerNorm respecto a gamma", n2.gamma(),
                       [&](Tensor&) { return (n2(fixed) * w).sum(); });
        check_gradient("gradiente de LayerNorm respecto a beta", n2.beta(),
                       [&](Tensor&) { return (n2(fixed) * w).sum(); });
    }

    // Embedding
    nn::Embedding emb(5, 3);
    check(emb.weight().shape() == std::vector<size_t>({5, 3}), "Embedding crea una tabla (vocab, dim)");

    Tensor ids({2, 3}, {0, 1, 2,
                        4, 4, 0}, false);
    Tensor vectors = emb(ids);
    check(vectors.shape() == std::vector<size_t>({2, 3, 3}), "Embedding da (batch, seq, dim)");
    check_close(vectors.data()[0], emb.weight().data()[0], "busca la fila correcta del token 0");
    check_close(vectors.data()[9], emb.weight().data()[12], "busca la fila correcta del token 4");

    check_throws([&] { emb(Tensor({3}, 1.0f)); }, "Embedding con indices 1D lanza excepcion");
    check_throws([&] { emb(Tensor({1, 2}, {0.0f, 9.0f})); },
                 "un token fuera del vocabulario lanza excepcion");

    // Un token repetido acumula gradiente en su fila
    emb.zero_grad();
    emb(ids).sum().backward();
    check_close(emb.weight().grad().data()[12], 2.0f, "el token 4 aparece 2 veces y acumula 2");
    check_close(emb.weight().grad().data()[0], 2.0f, "el token 0 tambien aparece 2 veces");
    check_close(emb.weight().grad().data()[9], 0.0f, "un token ausente no recibe gradiente");
}

void test_attention() {
    section("transformer: atencion");

    // Claves ortogonales: cada consulta debe recuperar su valor
    Tensor Q({3, 3}, {20, 0, 0,
                      0, 20, 0,
                      0, 0, 20}, false);
    Tensor K({3, 3}, {1, 0, 0,
                      0, 1, 0,
                      0, 0, 1}, false);
    Tensor V({3, 2}, {10, 100,
                      20, 200,
                      30, 300}, false);

    Tensor weights;
    Tensor out = nn::scaled_dot_product_attention(Q, K, V, nullptr, &weights);
    check(out.shape() == std::vector<size_t>({3, 2}), "la atencion da (seq, d_v)");
    check(weights.shape() == std::vector<size_t>({3, 3}), "los pesos son (seq, seq)");
    check_close(out.data()[0], 10.0f, "la consulta 0 recupera el valor 0", 0.5f);
    check_close(out.data()[3], 200.0f, "la consulta 1 recupera el valor 1", 5.0f);

    float row_sum = weights.data()[0] + weights.data()[1] + weights.data()[2];
    check_close(row_sum, 1.0f, "cada fila de pesos suma 1");

    // Mascara causal
    Tensor mask = nn::causal_mask(4);
    check(mask.shape() == std::vector<size_t>({4, 4}), "causal_mask es (seq, seq)");
    check_close(mask.data()[0], 0.0f, "la diagonal no esta enmascarada");
    check(mask.data()[1] < -1e8f, "la parte superior esta enmascarada");
    check_close(mask.data()[4], 0.0f, "la parte inferior no esta enmascarada");

    Tensor mq = Tensor::randn({4, 3});
    Tensor mw;
    nn::scaled_dot_product_attention(mq, mq, mq, &mask, &mw);
    check_close(mw.data()[0], 1.0f, "la primera posicion solo se atiende a si misma");
    check_close(mw.data()[1], 0.0f, "una posicion no atiende al futuro");
    check_close(mw.data()[4] + mw.data()[5], 1.0f, "la segunda posicion reparte entre 2 tokens");

    check_throws([&] {
        nn::scaled_dot_product_attention(Tensor({3, 4}, 1.0f), Tensor({3, 5}, 1.0f), V);
    }, "query y key con d_k distinto lanzan excepcion");

    // Gradiente de la atencion
    Tensor gq = Tensor::randn({2, 4, 3});
    Tensor k_fixed = Tensor::randn({2, 4, 3});
    Tensor v_fixed = Tensor::randn({2, 4, 3});
    Tensor w_attn = Tensor::randn({2, 4, 3});
    Tensor causal = nn::causal_mask(4);

    check_gradient("gradiente de la atencion respecto a la consulta", gq, [&](Tensor& t) {
        return (nn::scaled_dot_product_attention(t, k_fixed, v_fixed) * w_attn).sum();
    });
    check_gradient("gradiente de la atencion respecto a los valores", v_fixed, [&](Tensor& t) {
        return (nn::scaled_dot_product_attention(gq, k_fixed, t) * w_attn).sum();
    });
    check_gradient("gradiente de la atencion enmascarada", gq, [&](Tensor& t) {
        return (nn::scaled_dot_product_attention(t, t, t, &causal) * w_attn).sum();
    });
}

void test_positional_encoding() {
    section("transformer: codificacion posicional");

    Tensor pe = nn::positional_encoding(4, 6);
    check(pe.shape() == std::vector<size_t>({4, 6}), "positional_encoding da (seq, d_model)");

    // Posicion 0: sin(0) = 0 en los indices pares, cos(0) = 1 en los impares
    check_close(pe(0, 0), 0.0f, "PE[0] empieza con sin(0) == 0");
    check_close(pe(0, 1), 1.0f, "PE[0] sigue con cos(0) == 1");
    check_close(pe(1, 0), std::sin(1.0f), "PE[1][0] == sin(1)");
    check_close(pe(1, 1), std::cos(1.0f), "PE[1][1] == cos(1)");

    // Las frecuencias decrecen: la ultima pareja varia mucho menos con la posicion
    const float fast = std::fabs(pe(3, 0) - pe(0, 0));
    const float slow = std::fabs(pe(3, 4) - pe(0, 4));
    check(slow < fast, "las dimensiones altas usan frecuencias mas bajas");

    // Determinista y sin gradiente
    Tensor pe2 = nn::positional_encoding(4, 6);
    check_close(pe2(2, 3), pe(2, 3), "positional_encoding es determinista");
    check(!pe.requires_grad(), "la codificacion posicional no es un parametro entrenable");

    check_throws([&] { nn::positional_encoding(0, 4); }, "una longitud nula lanza excepcion");
}

void test_multihead_and_block() {
    section("transformer: MultiHeadAttention y TransformerBlock");

    engine::manual_seed(29);

    nn::MultiHeadAttention mha(8, 2);
    Tensor x = Tensor::randn({2, 5, 8});
    Tensor out = mha(x);
    check(out.shape() == std::vector<size_t>({2, 5, 8}), "MultiHeadAttention conserva (B, S, d_model)");
    check(mha.num_parameters() == 4 * (8 * 8 + 8), "MHA tiene 4 proyecciones de d_model x d_model");
    check(mha.last_attention().shape() == std::vector<size_t>({2, 2, 5, 5}),
          "los pesos de atencion son (B, H, S, S)");
    check(!mha.last_attention().requires_grad(),
          "los pesos guardados estan desligados del grafo");

    check_throws([&] { nn::MultiHeadAttention(8, 3); },
                 "d_model no divisible entre las cabezas lanza excepcion");
    check_throws([&] { nn::MultiHeadAttention(8, 0); }, "cero cabezas lanza excepcion");
    check_throws([&] { mha(Tensor::randn({2, 5, 4})); },
                 "MHA con un d_model erroneo lanza excepcion");
    check_throws([&] { mha(Tensor::randn({5, 8})); }, "MHA con una entrada 2D lanza excepcion");

    // Cada cabeza atiende por separado: con 1 cabeza los pesos son (B,1,S,S)
    nn::MultiHeadAttention single(8, 1);
    single(x);
    check(single.last_attention().shape() == std::vector<size_t>({2, 1, 5, 5}),
          "con una sola cabeza los pesos son (B, 1, S, S)");

    Tensor gx = Tensor::randn({2, 3, 8});
    Tensor w_mha = Tensor::randn({2, 3, 8});
    check_gradient("gradiente de MultiHeadAttention", gx, [&](Tensor& t) {
        return (mha(t) * w_mha).sum();
    });

    // TransformerBlock
    nn::TransformerBlock block(8, 2, 16);
    Tensor block_out = block(x);
    check(block_out.shape() == x.shape(), "TransformerBlock conserva la forma");
    check(block.num_parameters() ==
              4 * (8 * 8 + 8)      // atencion
              + 2 * (8 + 8)        // dos LayerNorm
              + (8 * 16 + 16)      // ff1
              + (16 * 8 + 8),      // ff2
          "TransformerBlock suma atencion, normalizaciones y red densa");

    check_throws([&] { nn::TransformerBlock(8, 2, 0); },
                 "una capa oculta nula lanza excepcion");

    Tensor gb = Tensor::randn({2, 3, 8});
    Tensor w_block = Tensor::randn({2, 3, 8});
    Tensor block_mask = nn::causal_mask(3);

    check_gradient("gradiente de TransformerBlock", gb, [&](Tensor& t) {
        return (block(t) * w_block).sum();
    });

    // Con mascara causal el gradiente debe seguir siendo correcto
    check_gradient("gradiente de TransformerBlock con mascara causal", gb, [&](Tensor& t) {
        return (block.forward(t, &block_mask) * w_block).sum();
    });
}

void test_transformer_training() {
    section("transformer: entrenamiento");

    engine::manual_seed(41);

    // Tarea minima que exige orden: la etiqueta es el primer token de la
    // secuencia, pero todas las secuencias contienen los mismos dos tokens.
    const size_t N = 40;
    const size_t seq = 4;
    Tensor X({N, seq}, 0.0f, false);
    std::vector<size_t> y(N, 0);

    for (size_t n = 0; n < N; ++n) {
        const size_t first = n % 2;
        X.data()[n * seq + 0] = static_cast<float>(first);
        X.data()[n * seq + 1] = static_cast<float>(1 - first);
        X.data()[n * seq + 2] = static_cast<float>(first);
        X.data()[n * seq + 3] = static_cast<float>(1 - first);
        y[n] = first;
    }

    nn::Embedding emb(2, 16);
    nn::TransformerBlock block(16, 2, 32);
    nn::Linear head(16, 2);
    Tensor pe = nn::positional_encoding(seq, 16);

    std::vector<Tensor> params;
    for (nn::Module* m : {static_cast<nn::Module*>(&emb),
                          static_cast<nn::Module*>(&block),
                          static_cast<nn::Module*>(&head)}) {
        std::vector<Tensor> sub = m->parameters();
        params.insert(params.end(), sub.begin(), sub.end());
    }
    optim::Adam opt(params, 0.02f);

    auto forward = [&](const Tensor& ids) {
        Tensor h = block(emb(ids) + pe);
        Tensor first = h.permute({1, 0, 2}).select_rows({0}).reshape({ids.shape()[0], 16});
        return head(first);
    };

    float first_loss = 0.0f;
    float last_loss = 0.0f;
    for (int epoch = 0; epoch < 60; ++epoch) {
        opt.zero_grad();
        Tensor loss = nn::cross_entropy_loss(forward(X), y);
        loss.backward();
        opt.step();
        if (epoch == 0) first_loss = loss.data()[0];
        last_loss = loss.data()[0];
    }

    check(last_loss < first_loss, "la perdida del transformer disminuye");
    check_close(nn::accuracy(forward(X), y), 1.0f, "el transformer aprende la tarea al 100%");
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
    test_repeated_backward();
    test_row_indexing_and_batches();
    test_no_grad_and_errors();
    test_graph_is_released();
    test_nn_layers();
    test_softmax_and_losses();
    test_optimizers();
    test_im2col();
    test_conv_layers();
    test_conv_gradients();
    test_cnn_training();
    test_nd_tensor_ops();
    test_layernorm_and_embedding();
    test_attention();
    test_positional_encoding();
    test_multihead_and_block();
    test_transformer_training();
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
