#include "test_support.hpp"

using namespace testing;

namespace {

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

    Tensor logits({4, 3},
                  {0.4f, -1.2f, 2.0f, 1.1f, 0.3f, -0.7f, -2.0f, 0.9f, 0.2f, 0.6f, 0.6f, -1.5f});
    std::vector<size_t> targets = {2, 0, 1, 1};
    check_gradient("gradiente de cross_entropy_loss()", logits,
                   [&](Tensor& t) { return nn::cross_entropy_loss(t, targets); });
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
    check_throws([&] { Tensor({4}, 1.0f)(0, 0); },
                 "A(fila, col) sobre un tensor 1D lanza excepcion");

    // Escalar a la izquierda
    Tensor t({2}, {1.0f, 2.0f});
    check_close((2.0f * t).data()[1], 4.0f, "2.0f * t multiplica por la izquierda");
    check_close((1.0f + t).data()[0], 2.0f, "1.0f + t suma por la izquierda");
    check_close((10.0f - t).data()[1], 8.0f, "10.0f - t resta por la izquierda");

    // select_rows
    Tensor X({4, 2}, {0, 0, 1, 1, 2, 2, 3, 3}, true);
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
    check_gradient("gradiente de select_rows()", G,
                   [](Tensor& t) { return t.select_rows({2, 0, 2}).sum(); });
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

}  // namespace

void run_autograd_tests() {
    test_autograd_scalar();
    test_autograd_numeric();
    test_repeated_backward();
    test_row_indexing_and_batches();
    test_no_grad_and_errors();
    test_graph_is_released();
}
