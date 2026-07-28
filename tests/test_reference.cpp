// Validación del motor contra PyTorch.
//
// La verificación numérica de gradientes (tests/test_*.cpp) demuestra que el
// motor es coherente consigo mismo. Esto demuestra algo más fuerte: que
// coincide con la implementación de referencia del sector.
//
// Los ficheros de tests/fixtures/ los genera tools/generate_reference.py con
// PyTorch y están commiteados, así que estas pruebas se ejecutan en CI sin
// necesitar Python. Cada uno trae las entradas, los pesos, el gradiente de
// salida usado, la salida esperada y todos los gradientes esperados.

#include "test_support.hpp"

#include "engine/serialize.hpp"

#include <map>
#include <string>

using namespace testing;

namespace {

#ifndef ENGINE_FIXTURE_DIR
#define ENGINE_FIXTURE_DIR "tests/fixtures"
#endif

// float32 acumula en distinto orden en PyTorch y aquí, así que la igualdad
// exacta no es alcanzable ni deseable. 1e-4 absoluto sobre valores del orden
// de la unidad deja margen de sobra para detectar una derivada mal escrita.
constexpr float kTolerance = 1e-4f;

using Fixture = std::map<std::string, Tensor>;

Fixture load(const std::string& name) {
    Fixture fixture;
    const std::string path = std::string(ENGINE_FIXTURE_DIR) + "/" + name + ".bin";
    for (auto& entry : engine::load_tensors(path)) {
        fixture.emplace(entry.first, entry.second);
    }
    return fixture;
}

const Tensor& get(const Fixture& fixture, const std::string& key) {
    auto it = fixture.find(key);
    if (it == fixture.end()) {
        throw std::runtime_error("El fixture no contiene '" + key + "'.");
    }
    return it->second;
}

// Compara contra PyTorch informando de la desviación máxima, que es el dato
// útil cuando algo falla: dice si es ruido de coma flotante o un error real.
void check_matches(const Tensor& actual, const Tensor& expected, const std::string& what,
                   float tolerance = kTolerance) {
    if (actual.shape() != expected.shape()) {
        check(false, what + " (forma " + actual.shape_str() + ", PyTorch da " +
                         expected.shape_str() + ")");
        return;
    }
    float max_diff = 0.0f;
    for (size_t i = 0; i < actual.size(); ++i) {
        max_diff = std::max(max_diff, std::fabs(actual.data()[i] - expected.data()[i]));
    }
    ++g_checks;
    if (max_diff <= tolerance) {
        std::cout << "  [ ok ] " << what << " (desviacion maxima " << std::scientific
                  << std::setprecision(2) << max_diff << std::defaultfloat << ")\n";
    } else {
        ++g_failures;
        std::cout << "  [FAIL] " << what << " (desviacion maxima " << max_diff
                  << " > " << tolerance << ")\n";
    }
}

// Copia los pesos del fixture en un parámetro del motor, conservando el
// requires_grad que la capa necesita para acumular gradiente.
void set_param(Tensor& param, const Tensor& source) {
    if (param.shape() != source.shape()) {
        throw std::runtime_error("El peso del fixture tiene forma " + source.shape_str() +
                                 " y la capa espera " + param.shape_str() + ".");
    }
    param.data() = source.data();
    param.set_requires_grad(true);
    param.zero_grad();
}

Tensor input_of(const Fixture& fixture, const std::string& key) {
    Tensor t = get(fixture, key).detach();
    t.set_requires_grad(true);
    return t;
}

// ---------------------------------------------------------
// Operaciones del tensor
// ---------------------------------------------------------

void test_matmul_cases() {
    section("PyTorch: multiplicacion matricial");

    for (const std::string& name : {"matmul", "matmul_batched", "matmul_shared"}) {
        Fixture f = load(name);
        Tensor a = input_of(f, "a");
        Tensor b = input_of(f, "b");

        Tensor out = a.matmul(b);
        check_matches(out, get(f, "output"), name + ": salida");

        out.backward(get(f, "grad_output"));
        check_matches(a.grad(), get(f, "grad.a"), name + ": gradiente de A");
        check_matches(b.grad(), get(f, "grad.b"), name + ": gradiente de B");
    }
}

void test_softmax_cases() {
    section("PyTorch: softmax");

    for (const std::string& name : {"softmax", "softmax_3d"}) {
        Fixture f = load(name);
        Tensor x = input_of(f, "input");

        Tensor out = x.softmax();
        check_matches(out, get(f, "output"), name + ": salida");

        out.backward(get(f, "grad_output"));
        check_matches(x.grad(), get(f, "grad.input"), name + ": gradiente");
    }
}

void test_activation_cases() {
    section("PyTorch: activaciones");

    nn::ReLU relu;
    nn::Sigmoid sigmoid;
    nn::Tanh tanh_layer;
    nn::GELU gelu;

    const std::pair<std::string, nn::Module*> cases[] = {
        {"act_relu", &relu}, {"act_sigmoid", &sigmoid},
        {"act_tanh", &tanh_layer}, {"act_gelu", &gelu},
    };

    for (const auto& entry : cases) {
        Fixture f = load(entry.first);
        Tensor x = input_of(f, "input");

        Tensor out = entry.second->forward(x);
        check_matches(out, get(f, "output"), entry.first + ": salida");

        out.backward(get(f, "grad_output"));
        check_matches(x.grad(), get(f, "grad.input"), entry.first + ": gradiente");
    }
}

void test_reduction_cases() {
    section("PyTorch: reducciones por eje");

    Fixture f = load("reductions");
    const std::tuple<std::string, size_t, const char*> cases[] = {
        {"sum0", 0, "sum"}, {"sum1", 1, "sum"}, {"mean1", 1, "mean"},
    };

    for (const auto& entry : cases) {
        const std::string tag = std::get<0>(entry);
        const size_t axis = std::get<1>(entry);
        const std::string op = std::get<2>(entry);

        Tensor x = input_of(f, "input");
        Tensor out = (op == "sum") ? x.sum(axis) : x.mean(axis);
        check_matches(out, get(f, "output." + tag), "reductions " + tag + ": salida");

        out.backward(get(f, "grad_output." + tag));
        check_matches(x.grad(), get(f, "grad." + tag), "reductions " + tag + ": gradiente");
    }

    Tensor xm = input_of(f, "input.max");
    Tensor out = xm.max(1);
    check_matches(out, get(f, "output.max1"), "reductions max1: salida");
    out.backward(get(f, "grad_output.max1"));
    check_matches(xm.grad(), get(f, "grad.max1"), "reductions max1: gradiente");
}

// ---------------------------------------------------------
// Capas
// ---------------------------------------------------------

void test_linear_cases() {
    section("PyTorch: capa densa");

    for (const std::string& name : {"linear", "linear_3d"}) {
        Fixture f = load(name);
        const size_t in_features = get(f, "weight").shape()[0];
        const size_t out_features = get(f, "weight").shape()[1];

        nn::Linear layer(in_features, out_features);
        set_param(layer.weight(), get(f, "weight"));
        set_param(layer.bias(), get(f, "bias"));

        Tensor x = input_of(f, "input");
        Tensor out = layer(x);
        check_matches(out, get(f, "output"), name + ": salida");

        out.backward(get(f, "grad_output"));
        check_matches(x.grad(), get(f, "grad.input"), name + ": gradiente de la entrada");
        check_matches(layer.weight().grad(), get(f, "grad.weight"), name + ": gradiente del peso");
        check_matches(layer.bias().grad(), get(f, "grad.bias"), name + ": gradiente del sesgo");
    }
}

void test_conv_cases() {
    section("PyTorch: convolucion y submuestreo");

    const std::tuple<std::string, size_t, size_t, size_t, size_t> cases[] = {
        {"conv2d_pad1", 3, 4, 1, 1},     // in, out, stride, padding
        {"conv2d_stride2", 1, 2, 2, 0},
    };

    for (const auto& entry : cases) {
        const std::string name = std::get<0>(entry);
        Fixture f = load(name);

        nn::Conv2d conv(std::get<1>(entry), std::get<2>(entry),
                        nn::Window2d(3, 3, std::get<3>(entry), std::get<4>(entry)));
        set_param(conv.weight(), get(f, "weight"));
        set_param(conv.bias(), get(f, "bias"));

        Tensor x = input_of(f, "input");
        Tensor out = conv(x);
        check_matches(out, get(f, "output"), name + ": salida");

        out.backward(get(f, "grad_output"));
        check_matches(x.grad(), get(f, "grad.input"), name + ": gradiente de la entrada");
        check_matches(conv.weight().grad(), get(f, "grad.weight"), name + ": gradiente del kernel");
        check_matches(conv.bias().grad(), get(f, "grad.bias"), name + ": gradiente del sesgo");
    }

    Fixture f = load("maxpool2d");
    nn::MaxPool2d pool(2, 2);
    Tensor x = input_of(f, "input");
    Tensor out = pool(x);
    check_matches(out, get(f, "output"), "maxpool2d: salida");
    out.backward(get(f, "grad_output"));
    check_matches(x.grad(), get(f, "grad.input"), "maxpool2d: gradiente");
}

void test_layernorm_case() {
    section("PyTorch: LayerNorm");

    Fixture f = load("layernorm");
    nn::LayerNorm norm(8);
    set_param(norm.gamma(), get(f, "gamma"));
    set_param(norm.beta(), get(f, "beta"));

    Tensor x = input_of(f, "input");
    Tensor out = norm(x);
    check_matches(out, get(f, "output"), "layernorm: salida");

    out.backward(get(f, "grad_output"));
    check_matches(x.grad(), get(f, "grad.input"), "layernorm: gradiente de la entrada");
    check_matches(norm.gamma().grad(), get(f, "grad.gamma"), "layernorm: gradiente de gamma");
    check_matches(norm.beta().grad(), get(f, "grad.beta"), "layernorm: gradiente de beta");
}

void test_cross_entropy_case() {
    section("PyTorch: entropia cruzada");

    Fixture f = load("cross_entropy");
    Tensor logits = input_of(f, "logits");

    std::vector<size_t> targets;
    // El Tensor se mantiene vivo en una variable: recorrer directamente
    // get(f, "targets").data() dejaría la referencia colgando, porque el
    // temporal muere antes de que empiece el bucle.
    const Tensor target_values = get(f, "targets");
    for (float v : target_values.data()) targets.push_back(static_cast<size_t>(v));

    Tensor loss = nn::cross_entropy_loss(logits, targets);
    check_matches(loss, get(f, "output"), "cross_entropy: perdida");

    loss.backward();
    check_matches(logits.grad(), get(f, "grad.logits"), "cross_entropy: gradiente");
}

// ---------------------------------------------------------
// Atención
// ---------------------------------------------------------

void test_attention_cases() {
    section("PyTorch: atencion");

    for (const std::string& name : {"attention", "attention_causal"}) {
        Fixture f = load(name);
        Tensor q = input_of(f, "q");
        Tensor k = input_of(f, "k");
        Tensor v = input_of(f, "v");

        const bool causal = (name == "attention_causal");
        Tensor mask;
        if (causal) mask = get(f, "mask");

        Tensor weights;
        Tensor out = nn::scaled_dot_product_attention(q, k, v, causal ? &mask : nullptr, &weights);

        check_matches(out, get(f, "output"), name + ": salida");
        check_matches(weights, get(f, "weights"), name + ": pesos de atencion");

        out.backward(get(f, "grad_output"));
        check_matches(q.grad(), get(f, "grad.q"), name + ": gradiente de Q");
        check_matches(k.grad(), get(f, "grad.k"), name + ": gradiente de K");
        check_matches(v.grad(), get(f, "grad.v"), name + ": gradiente de V");
    }
}

void test_multihead_case() {
    section("PyTorch: atencion multi-cabeza");

    Fixture f = load("multihead_attention");
    nn::MultiHeadAttention mha(8, 2);

    // parameters() devuelve las cuatro proyecciones en orden: query, key,
    // value y salida, cada una con su peso y su sesgo.
    std::vector<Tensor> params = mha.parameters();
    const char* tags[] = {"query", "key", "value", "out"};
    for (size_t i = 0; i < 4; ++i) {
        set_param(params[i * 2], get(f, std::string("w.") + tags[i] + ".weight"));
        set_param(params[i * 2 + 1], get(f, std::string("w.") + tags[i] + ".bias"));
    }

    Tensor x = input_of(f, "input");
    Tensor out = mha(x);
    check_matches(out, get(f, "output"), "multihead: salida");

    out.backward(get(f, "grad_output"));
    check_matches(x.grad(), get(f, "grad.input"), "multihead: gradiente de la entrada");
    for (size_t i = 0; i < 4; ++i) {
        check_matches(params[i * 2].grad(), get(f, std::string("grad.") + tags[i] + ".weight"),
                      std::string("multihead: gradiente del peso ") + tags[i]);
        check_matches(params[i * 2 + 1].grad(), get(f, std::string("grad.") + tags[i] + ".bias"),
                      std::string("multihead: gradiente del sesgo ") + tags[i]);
    }
}

void test_transformer_block_case() {
    section("PyTorch: bloque Transformer completo");

    Fixture f = load("transformer_block");
    nn::TransformerBlock block(8, 2, 16);

    // parameters() recorre norm1, norm2, la atencion (query, key, value,
    // salida) y las dos capas densas, en ese orden.
    std::vector<Tensor> params = block.parameters();
    const char* names[] = {"norm1.gamma", "norm1.beta", "norm2.gamma", "norm2.beta",
                           "query.weight", "query.bias", "key.weight", "key.bias",
                           "value.weight", "value.bias", "out.weight", "out.bias",
                           "ff1.weight", "ff1.bias", "ff2.weight", "ff2.bias"};
    check(params.size() == 16, "el bloque expone los 16 parametros esperados");

    for (size_t i = 0; i < params.size(); ++i) {
        set_param(params[i], get(f, std::string("w.") + names[i]));
    }

    Tensor x = input_of(f, "input");
    Tensor out = block(x);
    check_matches(out, get(f, "output"), "transformer_block: salida");

    out.backward(get(f, "grad_output"));
    check_matches(x.grad(), get(f, "grad.input"), "transformer_block: gradiente de la entrada");
    for (size_t i = 0; i < params.size(); ++i) {
        check_matches(params[i].grad(), get(f, std::string("grad.") + names[i]),
                      std::string("transformer_block: gradiente de ") + names[i]);
    }
}

// ---------------------------------------------------------
// Optimizadores
// ---------------------------------------------------------

// Se compara la trayectoria completa, no solo el resultado: un error en la
// corrección de sesgo de Adam puede dar el primer paso correcto y desviarse
// después.
void test_optimizer_trajectories() {
    section("PyTorch: trayectorias de los optimizadores");

    {
        Fixture f = load("adam");
        Tensor w = input_of(f, "initial");
        Tensor target = get(f, "target");
        optim::Adam opt({w}, 0.1f);

        for (size_t step = 0; step < 10; ++step) {
            opt.zero_grad();
            Tensor diff = w - target;
            (diff * diff).mean().backward();
            opt.step();
            check_matches(w, get(f, "step." + std::to_string(step)),
                          "Adam: paso " + std::to_string(step), 2e-4f);
        }
    }
    {
        Fixture f = load("sgd_momentum");
        Tensor w = input_of(f, "initial");
        Tensor target = get(f, "target");
        optim::SGD opt({w}, 0.05f, 0.9f);

        for (size_t step = 0; step < 10; ++step) {
            opt.zero_grad();
            Tensor diff = w - target;
            (diff * diff).mean().backward();
            opt.step();
            check_matches(w, get(f, "step." + std::to_string(step)),
                          "SGD con momento: paso " + std::to_string(step), 2e-4f);
        }
    }
}

} // namespace

void run_reference_tests() {
    test_matmul_cases();
    test_softmax_cases();
    test_activation_cases();
    test_reduction_cases();
    test_linear_cases();
    test_conv_cases();
    test_layernorm_case();
    test_cross_entropy_case();
    test_attention_cases();
    test_multihead_case();
    test_transformer_block_case();
    test_optimizer_trajectories();
}
