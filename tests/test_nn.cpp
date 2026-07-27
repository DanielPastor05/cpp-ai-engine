#include "test_support.hpp"

#include "engine/serialize.hpp"

#include <cstdio>
#include <fstream>
#include <set>

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


void test_activations() {
    section("nn: activaciones nuevas");

    Tensor x({5}, {-2.0f, -0.5f, 0.0f, 0.5f, 2.0f});

    nn::Sigmoid sig;
    Tensor s = sig(x);
    check_close(s.data()[2], 0.5f, "sigmoid(0) == 0.5");
    check(s.data()[0] > 0.0f && s.data()[0] < 0.5f, "sigmoid de un negativo cae en (0, 0.5)");
    check(s.data()[4] > 0.5f && s.data()[4] < 1.0f, "sigmoid de un positivo cae en (0.5, 1)");
    // Estabilidad en los extremos: sin el calculo por ramas, exp() desbordaria
    Tensor extreme({2}, {-100.0f, 100.0f});
    check_close(sig(extreme).data()[0], 0.0f, "sigmoid es estable en -100");
    check_close(sig(extreme).data()[1], 1.0f, "sigmoid es estable en +100");

    nn::Tanh th;
    check_close(th(x).data()[2], 0.0f, "tanh(0) == 0");
    check_close(th(x).data()[4], std::tanh(2.0f), "tanh(2) coincide con la libreria");

    nn::GELU gelu;
    Tensor g = gelu(x);
    check_close(g.data()[2], 0.0f, "gelu(0) == 0");
    check(g.data()[0] < 0.0f, "gelu deja pasar algo de senal negativa");
    check(g.data()[0] > -0.2f, "pero muy atenuada");
    check(g.data()[4] > 1.9f, "gelu es casi la identidad para positivos grandes");

    // Gradientes
    Tensor G({4}, {-1.5f, -0.3f, 0.7f, 2.1f});
    Tensor w = Tensor::randn({4});
    check_gradient("gradiente de Sigmoid", G, [&](Tensor& t) { return (sig(t) * w).sum(); });
    check_gradient("gradiente de Tanh", G, [&](Tensor& t) { return (th(t) * w).sum(); });
    check_gradient("gradiente de GELU", G, [&](Tensor& t) { return (gelu(t) * w).sum(); });
}

void test_train_eval_and_dropout() {
    section("nn: modo train/eval y Dropout");

    engine::manual_seed(77);

    nn::Dropout drop(0.5f);
    check(drop.is_training(), "los modulos arrancan en modo entrenamiento");

    Tensor x({1000}, 1.0f, false);
    Tensor trained = drop(x);
    size_t zeros = 0;
    for (float v : trained.data()) if (v == 0.0f) ++zeros;
    check(zeros > 400 && zeros < 600, "en entrenamiento anula alrededor de la mitad");

    // La media se conserva gracias al escalado 1/(1-p)
    float mean = 0.0f;
    for (float v : trained.data()) mean += v;
    mean /= 1000.0f;
    check(std::fabs(mean - 1.0f) < 0.1f, "el escalado conserva la media");

    drop.eval();
    check(!drop.is_training(), "eval() apaga el modo entrenamiento");
    Tensor evaluated = drop(x);
    bool identical = true;
    for (size_t i = 0; i < evaluated.size(); ++i) {
        if (evaluated.data()[i] != 1.0f) identical = false;
    }
    check(identical, "en evaluacion Dropout es la identidad");

    check_throws([&] { nn::Dropout(1.0f); }, "una probabilidad de 1 lanza excepcion");
    check_throws([&] { nn::Dropout(-0.1f); }, "una probabilidad negativa lanza excepcion");

    // El interruptor se propaga por el contenedor
    nn::Sequential model{
        nn::make<nn::Linear>(4, 4),
        nn::make<nn::Dropout>(0.5f),
        nn::make<nn::Linear>(4, 2)
    };
    model.eval();
    check(!model.at(1).is_training(), "Sequential propaga eval() a sus capas");
    model.train();
    check(model.at(1).is_training(), "Sequential propaga train() a sus capas");

    // El gradiente atraviesa solo las posiciones que sobrevivieron
    engine::manual_seed(5);
    nn::Dropout d2(0.5f);
    Tensor gx({20}, 1.0f, true);
    Tensor out = d2(gx);
    out.sum().backward();
    bool coherent = true;
    for (size_t i = 0; i < gx.size(); ++i) {
        const bool alive = out.data()[i] != 0.0f;
        if (alive != (gx.grad().data()[i] != 0.0f)) coherent = false;
    }
    check(coherent, "el gradiente pasa exactamente por donde paso la activacion");
}

void test_named_parameters() {
    section("nn: parametros con nombre");

    nn::Sequential model{
        nn::make<nn::Linear>(3, 4),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(4, 2)
    };

    auto named = model.named_parameters();
    check(named.size() == model.parameters().size(),
          "hay un nombre por cada parametro");

    std::set<std::string> unique;
    for (const auto& entry : named) unique.insert(entry.first);
    check(unique.size() == named.size(), "los nombres no se repiten");

    // El indice de la capa entra en el nombre, asi que dos capas iguales
    // no colisionan
    check(named[0].first != named[2].first, "dos capas Linear iguales tienen nombres distintos");

    // Comparten implementacion con los pesos reales
    named[0].second.data()[0] = 42.0f;
    check_close(model.parameters()[0].data()[0], 42.0f,
                "los tensores con nombre comparten datos con la capa");
}

void test_clip_and_schedulers() {
    section("optim: recorte de gradiente y planificadores");

    // Recorte
    Tensor a({2}, {3.0f, 4.0f}, true);   // norma 5
    (a * 1.0f).sum().backward();          // gradiente (1, 1) -> norma sqrt(2)
    float norm = optim::clip_grad_norm({a}, 10.0f);
    check_close(norm, std::sqrt(2.0f), "clip devuelve la norma previa");
    check_close(a.grad().data()[0], 1.0f, "por debajo del limite no recorta");

    Tensor b({2}, {1.0f, 1.0f}, true);
    b.add_grad(Tensor({2}, {3.0f, 4.0f}));  // norma 5
    float n2 = optim::clip_grad_norm({b}, 1.0f);
    check_close(n2, 5.0f, "clip mide la norma global");
    float after = std::sqrt(b.grad().data()[0] * b.grad().data()[0] +
                            b.grad().data()[1] * b.grad().data()[1]);
    check_close(after, 1.0f, "tras recortar la norma es el maximo", 1e-3f);
    check_close(b.grad().data()[0] / b.grad().data()[1], 3.0f / 4.0f,
                "el recorte conserva la direccion", 1e-3f);

    // La norma es global, no por parametro
    Tensor p1({1}, std::vector<float>{0.0f}, true);
    Tensor p2({1}, std::vector<float>{0.0f}, true);
    p1.add_grad(Tensor({1}, std::vector<float>{3.0f}));
    p2.add_grad(Tensor({1}, std::vector<float>{4.0f}));
    check_close(optim::clip_grad_norm({p1, p2}, 100.0f), 5.0f,
                "la norma junta de dos parametros es 5");

    check_throws([&] { optim::clip_grad_norm({a}, 0.0f); },
                 "un max_norm no positivo lanza excepcion");

    // Planificadores
    Tensor w({1}, std::vector<float>{0.0f}, true);
    optim::SGD opt({w}, 1.0f);

    optim::StepLR step(opt, 2, 0.5f);
    check_close(opt.learning_rate(), 1.0f, "el learning rate arranca en el base");
    step.step();
    check_close(opt.learning_rate(), 1.0f, "StepLR mantiene el lr dentro del escalon");
    step.step();
    check_close(opt.learning_rate(), 0.5f, "StepLR lo multiplica por gamma al cambiar de escalon");
    step.step(); step.step();
    check_close(opt.learning_rate(), 0.25f, "StepLR acumula los escalones");

    optim::SGD opt2({w}, 1.0f);
    optim::CosineAnnealingLR cos(opt2, 10, 0.0f);
    for (int i = 0; i < 5; ++i) cos.step();
    check_close(opt2.learning_rate(), 0.5f, "a mitad del coseno el lr es la mitad", 1e-3f);
    for (int i = 0; i < 5; ++i) cos.step();
    check_close(opt2.learning_rate(), 0.0f, "al final del coseno llega al minimo", 1e-3f);

    optim::SGD opt3({w}, 1.0f);
    optim::WarmupCosineLR warm(opt3, 3, 10, 0.0f);
    warm.step();
    check(opt3.learning_rate() < 0.5f, "durante el calentamiento el lr es bajo");
    warm.step(); warm.step();
    check_close(opt3.learning_rate(), 1.0f, "al terminar el calentamiento llega al base", 1e-3f);
    for (int i = 0; i < 7; ++i) warm.step();
    check(opt3.learning_rate() < 0.01f, "despues desciende en coseno");

    check_throws([&] { optim::StepLR(opt, 0); }, "un escalon nulo lanza excepcion");
    check_throws([&] { optim::WarmupCosineLR(opt, 10, 5); },
                 "un calentamiento mas largo que el total lanza excepcion");
}

void test_serialization() {
    section("serialize: guardar y cargar pesos");

    const std::string path = "test_weights.bin";
    engine::manual_seed(99);

    nn::Sequential model{
        nn::make<nn::Linear>(4, 6),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(6, 3)
    };
    Tensor x = Tensor::randn({5, 4});
    Tensor before = model(x);

    engine::save_parameters(model, path);

    // Otro modelo con la misma arquitectura pero pesos distintos
    engine::manual_seed(1234);
    nn::Sequential loaded{
        nn::make<nn::Linear>(4, 6),
        nn::make<nn::ReLU>(),
        nn::make<nn::Linear>(6, 3)
    };
    Tensor different = loaded(x);
    bool differs = false;
    for (size_t i = 0; i < before.size(); ++i) {
        if (std::fabs(before.data()[i] - different.data()[i]) > 1e-5f) differs = true;
    }
    check(differs, "antes de cargar, los dos modelos dan salidas distintas");

    const size_t n = engine::load_parameters(loaded, path);
    check(n == model.parameters().size(), "se cargan todos los parametros");

    Tensor after = loaded(x);
    float max_diff = 0.0f;
    for (size_t i = 0; i < before.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(before.data()[i] - after.data()[i]));
    }
    check_close(max_diff, 0.0f, "tras cargar, el modelo reproduce la salida exacta");

    // Inspeccionar sin cargar
    auto summary = engine::inspect_parameters(path);
    check(summary.size() == model.parameters().size(), "inspect enumera todos los tensores");
    check(summary[0].second == std::vector<size_t>({4, 6}), "inspect devuelve las formas");

    // Una arquitectura distinta se rechaza en vez de cargarse mal
    nn::Sequential wrong{ nn::make<nn::Linear>(4, 8) };
    check_throws([&] { engine::load_parameters(wrong, path); },
                 "cargar en una arquitectura distinta lanza excepcion");

    // Con strict=false se cargan solo los que casan por nombre. Como el nombre
    // por defecto incluye las dimensiones de la capa, una arquitectura distinta
    // simplemente no casa con nada.
    check(engine::load_parameters(wrong, path, false) == 0,
          "sin strict, una arquitectura distinta no carga nada");

    // La comprobacion de forma salta cuando el nombre SI coincide pero la
    // forma no: es el caso que dejaria un modelo roto en silencio.
    {
        Tensor good({2, 2}, 1.0f, true);
        std::vector<std::pair<std::string, Tensor>> saved = {{"peso", good}};
        engine::save_parameters(saved, "shape_test.bin");

        Tensor mismatched({3, 3}, 0.0f, true);
        std::vector<std::pair<std::string, Tensor>> target = {{"peso", mismatched}};
        check_throws([&] { engine::load_parameters(target, "shape_test.bin"); },
                     "el mismo nombre con otra forma se rechaza");
        std::remove("shape_test.bin");
    }

    // Dos parametros con el mismo nombre serian indistinguibles al cargar
    {
        Tensor t1({1}, std::vector<float>{1.0f}, true);
        Tensor t2({1}, std::vector<float>{2.0f}, true);
        std::vector<std::pair<std::string, Tensor>> dup = {{"a", t1}, {"a", t2}};
        check_throws([&] { engine::save_parameters(dup, "dup.bin"); },
                     "nombres duplicados se rechazan al guardar");
        std::remove("dup.bin");
    }

    check_throws([&] { engine::load_parameters(model, "no_existe.bin"); },
                 "cargar un fichero inexistente lanza excepcion");

    // Un fichero que no es nuestro
    {
        std::ofstream bad("not_weights.bin", std::ios::binary);
        bad << "esto no son pesos en absoluto";
    }
    check_throws([&] { engine::load_parameters(model, "not_weights.bin"); },
                 "un fichero con firma incorrecta se rechaza");

    // Un transformer completo tambien va y vuelve
    engine::manual_seed(3);
    nn::TransformerBlock block(8, 2, 16);
    Tensor seq = Tensor::randn({2, 4, 8});
    Tensor block_before = block(seq);
    engine::save_parameters(block, path);

    engine::manual_seed(444);
    nn::TransformerBlock block2(8, 2, 16);
    engine::load_parameters(block2, path);
    Tensor block_after = block2(seq);
    float block_diff = 0.0f;
    for (size_t i = 0; i < block_before.size(); ++i) {
        block_diff = std::max(block_diff, std::fabs(block_before.data()[i] - block_after.data()[i]));
    }
    check_close(block_diff, 0.0f, "un TransformerBlock completo se guarda y se restaura");

    std::remove(path.c_str());
    std::remove("not_weights.bin");
}

} // namespace

void run_nn_tests() {
    test_nn_layers();
    test_softmax_and_losses();
    test_optimizers();
    test_end_to_end_training();
    test_activations();
    test_train_eval_and_dropout();
    test_named_parameters();
    test_clip_and_schedulers();
    test_serialization();
}
