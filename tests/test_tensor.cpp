#include "test_support.hpp"

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

} // namespace

void run_tensor_tests() {
    test_tensor_basics();
    test_matmul();
    test_broadcast_add();
    test_nd_tensor_ops();
}
