#include "engine/tensor.hpp"
#include <iostream>

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Phase 1: the tensor library, demonstrated       \n";
    std::cout << "====================================================\n\n";

    using engine::Tensor;

    // 1. Creación e inspección de Tensores 2D (Matrices)
    std::cout << "--- 1. Creacion e inspeccion de pasadas (Strides) ---\n";
    Tensor A({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    A.print("Matriz A (2x3)");

    std::cout << "Elemento en A[1, 2]: " << A({1, 2}) << " (Esperado: 6.0000)\n\n";

    // 2. Operaciones Elemento a Elemento
    std::cout << "--- 2. Operaciones Elemento a Elemento ---\n";
    Tensor B = Tensor::ones({2, 3}) * 2.0f;
    B.print("Matriz B (2x3 de doses)");

    Tensor C = A + B;
    C.print("C = A + B");

    Tensor D = A * B;
    D.print("D = A * B (Producto Hadamard)");

    // 3. Función de Activación ReLU
    std::cout << "--- 3. Funciones de Activacion (ReLU) ---\n";
    Tensor X({2, 3}, {-2.0f, 0.5f, -1.0f, 3.0f, -0.1f, 4.2f});
    X.print("Input tensor X (with negative values)");

    Tensor X_relu = X.relu();
    X_relu.print("X.relu()");

    // 4. Multiplicación Matricial (MatMul)
    std::cout << "--- 4. Multiplicacion Matricial (MatMul) ---\n";
    Tensor M1({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Tensor M2({3, 2}, {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f});

    M1.print("M1 (2x3)");
    M2.print("M2 (3x2)");

    Tensor M_res = M1.matmul(M2);
    M_res.print("M_res = M1.matmul(M2) (2x2)");

    /*
      Cálculo esperado para M1 x M2:
      Fila 0, Col 0: 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
      Fila 0, Col 1: 1*8 + 2*1 + 3*3 = 8 + 2 + 9   = 19
      Fila 1, Col 0: 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
      Fila 1, Col 1: 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    */

    // 5. Reshape (Reconfiguración de dimensiones)
    std::cout << "--- 5. Reshape (Reconfigurar Dimensiones) ---\n";
    Tensor Flat = Tensor::rand({1, 12}, 0.0f, 10.0f);
    Flat.print("Tensor Vector 1D (1x12)");

    Tensor Reshaped = Flat.reshape({3, 4});
    Reshaped.print("Reshaped a Matriz (3x4)");

    std::cout << "Phase 1 complete.\n";
    return 0;
}
