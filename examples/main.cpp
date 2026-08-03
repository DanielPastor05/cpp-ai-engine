#include "engine/tensor.hpp"
#include <iostream>

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Phase 1: the tensor library, demonstrated       \n";
    std::cout << "====================================================\n\n";

    using engine::Tensor;

    // 1. Creating and inspecting 2D tensors (matrices)
    std::cout << "--- 1. Creating and inspecting strides ---\n";
    Tensor A({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    A.print("Matrix A (2x3)");

    std::cout << "Element at A[1, 2]: " << A({1, 2}) << " (expected 6.0000)\n\n";

    // 2. Element-wise operations
    std::cout << "--- 2. Element-wise operations ---\n";
    Tensor B = Tensor::ones({2, 3}) * 2.0f;
    B.print("Matrix B (2x3 of twos)");

    Tensor C = A + B;
    C.print("C = A + B");

    Tensor D = A * B;
    D.print("D = A * B (Hadamard product)");

    // 3. The ReLU activation
    std::cout << "--- 3. Activation functions (ReLU) ---\n";
    Tensor X({2, 3}, {-2.0f, 0.5f, -1.0f, 3.0f, -0.1f, 4.2f});
    X.print("Input tensor X (with negative values)");

    Tensor X_relu = X.relu();
    X_relu.print("X.relu()");

    // 4. Matrix multiplication (MatMul)
    std::cout << "--- 4. Matrix multiplication (MatMul) ---\n";
    Tensor M1({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    Tensor M2({3, 2}, {7.0f, 8.0f, 9.0f, 1.0f, 2.0f, 3.0f});

    M1.print("M1 (2x3)");
    M2.print("M2 (3x2)");

    Tensor M_res = M1.matmul(M2);
    M_res.print("M_res = M1.matmul(M2) (2x2)");

    /*
      Expected computation for M1 x M2:
      Row 0, col 0: 1*7 + 2*9 + 3*2 = 7 + 18 + 6 = 31
      Row 0, col 1: 1*8 + 2*1 + 3*3 = 8 + 2 + 9   = 19
      Row 1, col 0: 4*7 + 5*9 + 6*2 = 28 + 45 + 12 = 85
      Row 1, col 1: 4*8 + 5*1 + 6*3 = 32 + 5 + 18 = 55
    */

    // 5. Reshape (rearranging the dimensions)
    std::cout << "--- 5. Reshape (rearranging the dimensions) ---\n";
    Tensor Flat = Tensor::rand({1, 12}, 0.0f, 10.0f);
    Flat.print("Tensor Vector 1D (1x12)");

    Tensor Reshaped = Flat.reshape({3, 4});
    Reshaped.print("Reshaped to a matrix (3x4)");

    std::cout << "Phase 1 complete.\n";
    return 0;
}
