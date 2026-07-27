#include "test_support.hpp"

int main() {
    std::cout << "====================================================\n";
    std::cout << "  Suite de pruebas de cpp-ai-engine                 \n";
    std::cout << "====================================================\n";

    run_tensor_tests();
    run_autograd_tests();
    run_nn_tests();
    run_conv_tests();
    run_transformer_tests();
    run_reference_tests();

    std::cout << "\n====================================================\n";
    if (testing::g_failures == 0) {
        std::cout << "  TODAS LAS PRUEBAS PASARON (" << testing::g_checks << " comprobaciones)\n";
    } else {
        std::cout << "  " << testing::g_failures << " DE " << testing::g_checks
                  << " COMPROBACIONES FALLARON\n";
    }
    std::cout << "====================================================\n";

    return testing::g_failures == 0 ? 0 : 1;
}
