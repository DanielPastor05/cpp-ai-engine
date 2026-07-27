#include "test_support.hpp"

using namespace testing;

namespace {


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

} // namespace

void run_conv_tests() {
    test_im2col();
    test_conv_layers();
    test_conv_gradients();
    test_cnn_training();
}
