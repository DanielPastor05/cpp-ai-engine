#include "test_support.hpp"

#include "engine/parallel.hpp"

using namespace testing;

namespace {

void test_tensor_basics() {
    section("Tensor: forma, strides e indexacion");

    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});
    check(A.shape() == std::vector<size_t>({2, 3}), "la forma es (2, 3)");
    check(A.strides() == std::vector<size_t>({3, 1}), "los strides row-major son (3, 1)");
    check(A.size() == 6, "el tensor tiene 6 elementos");
    check_close(A({1, 2}), 6.0f, "A[1, 2] == 6");

    check_throws([&] { A({2, 0}); }, "indexar fuera de rango lanza excepcion");
    check_throws([&] { Tensor({2, 3}, {1.0f, 2.0f}); },
                 "datos de tamano incorrecto lanzan excepcion");
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
    check(bc.shape() == std::vector<size_t>({3, 4}),
          "difundir (1,3,4) sobre (3,4) conserva la forma");
    check_close(bc.data()[0], 3.0f, "difundir con un eje inicial de tamano 1 suma bien");
    bc.sum().backward();
    check_close(leading_one.grad().data()[0], 1.0f,
                "su gradiente es 1, no la suma de repeticiones inexistentes");

    // Un escalar tambien se difunde sobre cualquier forma
    Tensor scalar({1}, std::vector<float>{5.0f}, true);
    Tensor plus_scalar = X + scalar;
    check_close(plus_scalar.data()[0], 6.0f, "un tensor de un elemento se difunde como escalar");
    plus_scalar.sum().backward();
    check_close(scalar.grad().data()[0], 6.0f,
                "el escalar acumula el gradiente de los 6 elementos");
}

void test_nd_tensor_ops() {
    section("Tensor: operaciones N-dimensionales");

    // transpose intercambia los dos ultimos ejes en cualquier rango
    Tensor T3({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor Tt = T3.transpose();
    check(Tt.shape() == std::vector<size_t>({2, 3, 2}),
          "transpose 3D intercambia los dos ultimos ejes");
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
    Tensor A({2, 2, 3}, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});
    Tensor B({2, 3, 2}, {1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 1, 1});
    Tensor C = A.matmul(B);
    check(C.shape() == std::vector<size_t>({2, 2, 2}), "matmul por lotes da (B, M, N)");
    check_close(C.data()[0], 4.0f, "lote 0: 1*1 + 2*0 + 3*1 == 4");
    check_close(C.data()[1], 5.0f, "lote 0: 1*0 + 2*1 + 3*1 == 5");
    check_close(C.data()[4], 23.0f, "lote 1: 7*2 + 8*0 + 9*1 == 23");

    // Un operando 2D se comparte con todo el lote
    Tensor shared({3, 2}, {1, 0, 0, 1, 1, 1});
    Tensor Cs = A.matmul(shared);
    check(Cs.shape() == std::vector<size_t>({2, 2, 2}),
          "matmul con matriz 2D compartida da (B, M, N)");
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
    check(sum_bc.shape() == std::vector<size_t>({2, 3, 4}),
          "la difusion (3,4) sobre (2,3,4) funciona");
    check_close(sum_bc.data()[0], 3.0f, "la difusion suma el bloque repetido");
    sum_bc.sum().backward();
    check_close(row.grad().data()[0], 2.0f,
                "el gradiente del operando difundido suma las 2 repeticiones");

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
    check_gradient("gradiente de softmax() 3D", G,
                   [&](Tensor& t) { return (t.softmax() * w_soft).sum(); });
    check_gradient("gradiente de matmul por lotes", G, [](Tensor& t) {
        Tensor B2({2, 4, 2}, {1, -2, 0.5f, 3, -1, 2, 0.25f, -0.5f, 2, 1, -1, 0.5f, 3, -2, 1, 1});
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
    check_gradient("gradiente de Linear sobre una entrada 3D", seq_input,
                   [&](Tensor& t) { return (proj(t) * w_proj).sum(); });
}

void test_reductions() {
    section("Tensor: reducciones por eje");

    Tensor A({2, 3}, {1, 2, 3, 4, 5, 6});

    Tensor s0 = A.sum(0);
    check(s0.shape() == std::vector<size_t>({3}), "sum(0) elimina el primer eje");
    check_close(s0.data()[0], 5.0f, "sum(0) suma por columnas: 1 + 4");
    check_close(s0.data()[2], 9.0f, "sum(0) tercera columna: 3 + 6");

    Tensor s1 = A.sum(1);
    check(s1.shape() == std::vector<size_t>({2}), "sum(1) elimina el segundo eje");
    check_close(s1.data()[0], 6.0f, "sum(1) suma por filas: 1+2+3");

    Tensor sk = A.sum(1, true);
    check(sk.shape() == std::vector<size_t>({2, 1}), "keepdim deja el eje reducido a 1");

    Tensor m = A.mean(0);
    check_close(m.data()[0], 2.5f, "mean(0) promedia por columnas");

    Tensor mx = A.max(0);
    check(mx.shape() == std::vector<size_t>({3}), "max(0) elimina el primer eje");
    check_close(mx.data()[1], 5.0f, "max(0) toma el mayor de cada columna");

    // Reducir un tensor 1D deja un escalar
    Tensor v({4}, {1, 7, 3, 2});
    check(v.sum(0).shape() == std::vector<size_t>({1}), "reducir un 1D da un escalar {1}");
    check_close(v.max(0).data()[0], 7.0f, "max de un vector");

    check_throws([&] { A.sum(5); }, "reducir un eje inexistente lanza excepcion");
    check_throws([&] { A.max(2); }, "max sobre un eje inexistente lanza excepcion");

    // Gradientes
    Tensor G({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < G.size(); ++i) G.data()[i] = 0.37f * static_cast<float>(i) - 4.1f;
    Tensor w_sum = Tensor::randn({2, 4});
    check_gradient("gradiente de sum(axis)", G,
                   [&](Tensor& t) { return (t.sum(1) * w_sum).sum(); });
    check_gradient("gradiente de mean(axis)", G,
                   [&](Tensor& t) { return (t.mean(1) * w_sum).sum(); });
    // Valores bien separados: el maximo no es derivable en un empate
    Tensor Gm({2, 3, 4}, 0.0f);
    for (size_t i = 0; i < Gm.size(); ++i) Gm.data()[i] = static_cast<float>((i * 37) % 24) * 0.5f;
    check_gradient("gradiente de max(axis)", Gm,
                   [&](Tensor& t) { return (t.max(1) * w_sum).sum(); });

    // El gradiente del maximo va solo al ganador
    Tensor mg({1, 3}, {1.0f, 9.0f, 2.0f}, true);
    mg.max(1).sum().backward();
    check_close(mg.grad().data()[1], 1.0f, "el maximo recibe todo el gradiente");
    check_close(mg.grad().data()[0], 0.0f, "los demas no reciben nada");
}

void test_slice_concat_stack() {
    section("Tensor: slice, concat y stack");

    Tensor A({3, 4}, 0.0f);
    for (size_t i = 0; i < A.size(); ++i) A.data()[i] = static_cast<float>(i);

    Tensor r = A.slice(0, 1, 2);
    check(r.shape() == std::vector<size_t>({2, 4}), "slice sobre el primer eje");
    check_close(r(0, 0), 4.0f, "slice empieza en la fila pedida");

    Tensor c = A.slice(1, 1, 2);
    check(c.shape() == std::vector<size_t>({3, 2}), "slice sobre el segundo eje");
    check_close(c(0, 0), 1.0f, "slice de columnas toma la columna correcta");
    check_close(c(1, 1), 6.0f, "slice de columnas respeta las filas");

    check_throws([&] { A.slice(0, 2, 5); }, "un slice que se sale lanza excepcion");
    check_throws([&] { A.slice(0, 0, 0); }, "un slice vacio lanza excepcion");
    check_throws([&] { A.slice(7, 0, 1); }, "un slice sobre un eje inexistente lanza excepcion");

    // concat
    Tensor P({2, 2}, {1, 2, 3, 4});
    Tensor Q({1, 2}, {5, 6});
    Tensor cc = Tensor::concat({P, Q}, 0);
    check(cc.shape() == std::vector<size_t>({3, 2}), "concat sobre el primer eje suma las filas");
    check_close(cc(2, 0), 5.0f, "concat coloca la segunda parte detras");

    Tensor R({2, 3}, {7, 8, 9, 10, 11, 12});
    Tensor cc2 = Tensor::concat({P, R}, 1);
    check(cc2.shape() == std::vector<size_t>({2, 5}),
          "concat sobre el segundo eje suma las columnas");
    check_close(cc2(0, 2), 7.0f, "concat por columnas intercala correctamente");
    check_close(cc2(1, 0), 3.0f, "concat por columnas conserva las filas");

    check_throws([&] { Tensor::concat({}, 0); }, "concat sin partes lanza excepcion");
    check_throws([&] { Tensor::concat({P, Tensor({3, 3}, 1.0f)}, 0); },
                 "concat con dimensiones incompatibles lanza excepcion");

    // stack
    Tensor st = Tensor::stack({P, P}, 0);
    check(st.shape() == std::vector<size_t>({2, 2, 2}), "stack crea un eje nuevo");
    Tensor st1 = Tensor::stack({P, P}, 1);
    check(st1.shape() == std::vector<size_t>({2, 2, 2}), "stack admite el eje intermedio");
    check_throws([&] { Tensor::stack({P, Q}, 0); }, "stack con formas distintas lanza excepcion");

    // Gradientes
    Tensor G({3, 4}, 0.0f);
    for (size_t i = 0; i < G.size(); ++i) G.data()[i] = 0.31f * static_cast<float>(i) - 2.0f;
    Tensor w_sl = Tensor::randn({2, 4});
    check_gradient("gradiente de slice()", G,
                   [&](Tensor& t) { return (t.slice(0, 1, 2) * w_sl).sum(); });

    Tensor other({2, 4}, 1.5f);
    Tensor w_cc = Tensor::randn({5, 4});
    check_gradient("gradiente de concat() (primera parte)", G,
                   [&](Tensor& t) { return (Tensor::concat({t, other}, 0) * w_cc).sum(); });
    check_gradient("gradiente de concat() (segunda parte)", other,
                   [&](Tensor& t) { return (Tensor::concat({G, t}, 0) * w_cc).sum(); });

    // Concatenar un tensor consigo mismo acumula en ambas franjas
    Tensor twice({1, 2}, {1.0f, 2.0f}, true);
    Tensor::concat({twice, twice}, 0).sum().backward();
    check_close(twice.grad().data()[0], 2.0f, "concatenar un tensor consigo mismo acumula");
}

void test_broadcast_all_operators() {
    section("Tensor: difusion en los cuatro operadores");

    Tensor X({2, 3}, {1, 2, 3, 4, 5, 6});
    Tensor row({3}, {1.0f, 2.0f, 4.0f}, true);

    check_close((X - row).data()[0], 0.0f, "la resta difunde: 1 - 1");
    check_close((X - row).data()[4], 3.0f, "la resta difunde en la segunda fila: 5 - 2");
    check_close((X * row).data()[2], 12.0f, "el producto difunde: 3 * 4");
    check_close((X / row).data()[1], 1.0f, "la division difunde: 2 / 2");

    // Un tensor de un elemento actua como escalar
    Tensor k({1}, std::vector<float>{10.0f}, true);
    check_close((X * k).data()[3], 40.0f, "un tensor {1} se difunde como escalar");

    check_throws([&] { X - Tensor({4}, 1.0f); },
                 "una resta con sufijo incompatible lanza excepcion");
    check_throws([&] { X* Tensor({5, 3}, 1.0f); }, "un producto incompatible lanza excepcion");

    // Gradientes del operando difundido
    Tensor base({2, 3}, {1, 2, 3, 4, 5, 6}, false);
    Tensor b1({3}, {1.0f, 2.0f, 4.0f}, true);
    (base * b1).sum().backward();
    check_close(b1.grad().data()[0], 5.0f,
                "producto difundido: el gradiente suma la columna (1+4)");

    Tensor b2({3}, {1.0f, 2.0f, 4.0f}, true);
    (base - b2).sum().backward();
    check_close(b2.grad().data()[0], -2.0f, "resta difundida: el gradiente es -1 por fila");

    Tensor G({2, 3}, 0.0f);
    for (size_t i = 0; i < G.size(); ++i) G.data()[i] = 0.5f * static_cast<float>(i) + 1.0f;
    Tensor d({3}, {2.0f, 3.0f, 4.0f});
    Tensor w = Tensor::randn({2, 3});
    check_gradient("gradiente de la resta difundida", G,
                   [&](Tensor& t) { return ((t - d) * w).sum(); });
    check_gradient("gradiente del producto difundido", G,
                   [&](Tensor& t) { return ((t * d) * w).sum(); });
    check_gradient("gradiente de la division difundida", G,
                   [&](Tensor& t) { return ((t / d) * w).sum(); });
    check_gradient("gradiente del divisor difundido", d,
                   [&](Tensor& t) { return ((G / t) * w).sum(); });
}

void test_parallelism() {
    section("parallel: reparto de trabajo");

    namespace par = engine::parallel;
    const size_t original = par::num_threads();

    check(par::num_threads() >= 1, "el pool arranca con al menos un hilo");
    check(!par::inside_parallel_region(), "el hilo principal no esta dentro de una region");

    // Cobertura: cada indice se visita exactamente una vez
    for (size_t threads : {size_t(1), size_t(2), size_t(4)}) {
        par::set_num_threads(threads);
        const size_t n = 100000;
        std::vector<int> visits(n, 0);
        par::parallel_for(n, 1000, [&](size_t from, size_t to) {
            for (size_t i = from; i < to; ++i) visits[i]++;
        });
        bool exactly_once = true;
        for (int v : visits)
            if (v != 1) exactly_once = false;
        check(exactly_once, "con " + std::to_string(threads) +
                                " hilos cada indice se visita exactamente una vez");
    }

    // Determinismo: el reparto por filas no altera el orden de acumulacion,
    // asi que el resultado debe ser identico BIT A BIT, no solo parecido.
    engine::manual_seed(7);
    Tensor A = Tensor::randn({200, 150});
    Tensor B = Tensor::randn({150, 120});
    Tensor E1 = Tensor::randn({400, 400});
    Tensor E2 = Tensor::randn({400, 400});

    par::set_num_threads(1);
    Tensor mm_serial = A.matmul(B);
    Tensor add_serial = E1 + E2;

    par::set_num_threads(4);
    Tensor mm_par = A.matmul(B);
    Tensor add_par = E1 + E2;

    bool identical = true;
    for (size_t i = 0; i < mm_serial.size(); ++i) {
        if (mm_serial.data()[i] != mm_par.data()[i]) identical = false;
    }
    check(identical, "matmul da un resultado identico bit a bit con 1 y con 4 hilos");

    identical = true;
    for (size_t i = 0; i < add_serial.size(); ++i) {
        if (add_serial.data()[i] != add_par.data()[i]) identical = false;
    }
    check(identical, "la suma da un resultado identico bit a bit con 1 y con 4 hilos");

    // Las regiones anidadas se ejecutan en linea: multiplicar hilos dentro de
    // un hilo solo anade contencion
    par::set_num_threads(4);
    bool nested_inline = true;
    par::parallel_for(10000, 100, [&](size_t, size_t) {
        if (!par::inside_parallel_region()) nested_inline = false;
        size_t calls = 0;
        par::parallel_for(10000, 100, [&](size_t, size_t) { ++calls; });
        if (calls != 1) nested_inline = false;  // un solo trozo = ejecutado en linea
    });
    check(nested_inline, "una region anidada se ejecuta en linea");

    // Una excepcion en un trabajador llega al hilo que reparte
    check_throws(
        [&] {
            par::parallel_for(100000, 1000, [](size_t from, size_t) {
                if (from > 0) throw std::runtime_error("fallo en un trabajador");
            });
        },
        "una excepcion en un trabajador se propaga al que reparte");

    // Un rango vacio no hace nada
    size_t calls = 0;
    par::parallel_for(0, 10, [&](size_t, size_t) { ++calls; });
    check(calls == 0, "un rango vacio no ejecuta el cuerpo");

    par::set_num_threads(1);
    check(par::num_threads() == 1, "set_num_threads(1) deja solo el hilo llamante");
    calls = 0;
    par::parallel_for(1000000, 1, [&](size_t, size_t) { ++calls; });
    check(calls == 1, "con un hilo el cuerpo se ejecuta una sola vez, en linea");

    par::set_num_threads(original);
}

}  // namespace

void run_tensor_tests() {
    test_tensor_basics();
    test_matmul();
    test_broadcast_add();
    test_nd_tensor_ops();
    test_reductions();
    test_slice_concat_stack();
    test_broadcast_all_operators();
    test_parallelism();
}
