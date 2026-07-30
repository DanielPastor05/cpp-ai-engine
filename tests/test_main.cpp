#include "test_support.hpp"

int main() {
    std::cout << "====================================================\n";
    std::cout << "  cpp-ai-engine test suite                          \n";
    std::cout << "====================================================\n";

    run_tensor_tests();
    run_autograd_tests();
    run_nn_tests();
    run_conv_tests();
    run_transformer_tests();
    run_reference_tests();
    run_cuda_indexing_tests();
    run_cuda_parity_tests();

    std::cout << "\n====================================================\n";
    if (testing::g_failures == 0) {
        std::cout << "  ALL TESTS PASSED (" << testing::g_checks << " checks)\n";
    } else {
        std::cout << "  " << testing::g_failures << " OF " << testing::g_checks
                  << " CHECKS FAILED\n";
    }
    std::cout << "====================================================\n";

    return testing::g_failures == 0 ? 0 : 1;
}
