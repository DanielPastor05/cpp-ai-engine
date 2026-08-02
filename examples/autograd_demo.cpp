#include "engine/tensor.hpp"
#include "engine/autograd.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Phase 2: autograd engine and backpropagation       \n";
    std::cout << "====================================================\n\n";

    using engine::Tensor;

    // ---------------------------------------------------------
    // 1. Analytic gradients checked against autograd
    // ---------------------------------------------------------
    std::cout << "--- 1. Derivacion Automatica de Funcion Escalar ---\n";
    std::cout << "Funcion: L = a * b + ReLU(a)\n";
    std::cout << "Values: a = 2.0, b = 3.0\n\n";

    Tensor a({1}, std::vector<float>{2.0f}, true);
    Tensor b({1}, std::vector<float>{3.0f}, true);

    Tensor prod = a * b;
    Tensor relu_a = a.relu();
    Tensor L = prod + relu_a;

    // Run backpropagation
    L.backward();

    std::cout << "Resultado L = " << L.data()[0] << " (Esperado: 2*3 + 2 = 8.0000)\n";
    std::cout << "Gradiente dL/da = " << a.grad().data()[0] << " (Esperado: b + 1 = 4.0000)\n";
    std::cout << "Gradiente dL/db = " << b.grad().data()[0] << " (Esperado: a = 2.0000)\n\n";

    // ---------------------------------------------------------
    // 2. Autograd through matrix multiplication (MatMul)
    // ---------------------------------------------------------
    std::cout << "--- 2. Autograd over matrices (MatMul) ---\n";
    Tensor A({2, 3}, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, true);

    Tensor B({3, 2}, {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f}, true);

    Tensor C = A.matmul(B);
    Tensor Loss = C.sum();

    Loss.backward();

    std::cout << "Loss (total sum of C) = " << Loss.data()[0] << "\n";
    std::cout << "Gradient dLoss/dA (should match ones(2,2) x B^T):\n";
    A.grad().print("dL/dA");

    std::cout << "Gradient dLoss/dB (should match A^T x ones(2,2)):\n";
    B.grad().print("dL/dB");

    // ---------------------------------------------------------
    // 3. Training a linear regression by gradient descent
    // ---------------------------------------------------------
    std::cout << "--- 3. Gradient-descent optimisation (linear regression) ---\n";
    std::cout << "Goal: learn y = W * x + b where W_true = 2.5, b_true = 1.0\n\n";

    // The parameters to optimise. W is (1, 1) so it can go through matmul, and bias
    // is a (1, 1) row vector broadcast over the batch's 4 rows.
    Tensor W({1, 1}, std::vector<float>{0.0f}, true);     // Inicializar peso en 0
    Tensor bias({1, 1}, std::vector<float>{0.0f}, true);  // Inicializar sesgo en 0

    // Training data (x and y)
    Tensor x({4, 1}, {1.0f, 2.0f, 3.0f, 4.0f}, false);
    Tensor y_target({4, 1}, {3.5f, 6.0f, 8.5f, 11.0f}, false);  // 2.5 * x + 1.0

    float learning_rate = 0.05f;

    std::cout << "Training starts (100 iterations):\n";
    for (int epoch = 1; epoch <= 100; ++epoch) {
        // 0. Clear the previous iteration's gradients (they accumulate)
        W.zero_grad();
        bias.zero_grad();

        // 1. Forward Pass: pred = x * W + bias
        Tensor pred = x.matmul(W) + bias;

        // 2. Compute Loss: MSE = mean((pred - y_target)^2)
        Tensor diff = pred - y_target;
        Tensor loss = (diff * diff).mean();

        // 3. Backward Pass
        loss.backward();

        // 4. Update Parameters (Gradient Descent). Sin construir grafo.
        engine::autograd::NoGradGuard no_grad;
        W.data()[0] -= learning_rate * W.grad().data()[0];
        bias.data()[0] -= learning_rate * bias.grad().data()[0];

        if (epoch % 20 == 0 || epoch == 1) {
            std::cout << "Epoch " << std::setw(3) << epoch << " | Loss = " << std::fixed
                      << std::setprecision(6) << loss.data()[0] << " | W = " << std::setprecision(4)
                      << W.data()[0] << " | b = " << bias.data()[0] << "\n";
        }
    }

    std::cout << "\nResultados Finales:\n";
    std::cout << "W aprendido: " << W.data()[0] << " (Objetivo: 2.5000)\n";
    std::cout << "b aprendido: " << bias.data()[0] << " (Objetivo: 1.0000)\n\n";

    std::cout << "Phase 2 (autograd engine) complete.\n";
    return 0;
}
