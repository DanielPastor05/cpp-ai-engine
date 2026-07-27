#include "test_support.hpp"

using namespace testing;

namespace {


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

void run_nn_tests() {
    test_nn_layers();
    test_softmax_and_losses();
    test_optimizers();
    test_end_to_end_training();
}
