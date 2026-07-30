#ifndef ENGINE_TEST_SUPPORT_HPP
#define ENGINE_TEST_SUPPORT_HPP

// Utilities shared by the test files.
//
// The bulk of the autograd verification is done by numerical gradient checking
// (centred differences), which is the standard way to catch a chain rule that
// was derived wrongly.

#include "engine/tensor.hpp"
#include "engine/random.hpp"
#include "engine/autograd.hpp"
#include "engine/nn.hpp"
#include "engine/conv.hpp"
#include "engine/transformer.hpp"
#include "engine/optim.hpp"

#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using engine::Tensor;
namespace nn = engine::nn;
namespace optim = engine::optim;

namespace testing {

inline int g_failures = 0;
inline int g_checks = 0;

inline void check(bool condition, const std::string& what) {
    ++g_checks;
    if (condition) {
        std::cout << "  [ ok ] " << what << "\n";
    } else {
        ++g_failures;
        std::cout << "  [FAIL] " << what << "\n";
    }
}

inline void check_close(float actual, float expected, const std::string& what, float tol = 1e-4f) {
    const bool ok = std::fabs(actual - expected) <= tol;
    ++g_checks;
    if (ok) {
        std::cout << "  [ ok ] " << what << "\n";
    } else {
        ++g_failures;
        std::cout << "  [FAIL] " << what << " (got " << actual << ", expected " << expected
                  << ")\n";
    }
}

inline void check_throws(const std::function<void()>& fn, const std::string& what) {
    ++g_checks;
    try {
        fn();
        ++g_failures;
        std::cout << "  [FAIL] " << what << " (threw nothing)\n";
    } catch (const std::exception&) {
        std::cout << "  [ ok ] " << what << "\n";
    }
}

inline std::string win_name(const nn::Window2d& w) {
    return "k=" + std::to_string(w.kernel_h) + "x" + std::to_string(w.kernel_w) +
           ", s=" + std::to_string(w.stride) + ", p=" + std::to_string(w.padding);
}

inline void section(const std::string& title) {
    std::cout << "\n== " << title << " ==\n";
}

// Checks the analytic gradient of `loss_fn` with respect to `input` against the
// centred-difference approximation: (f(x+h) - f(x-h)) / 2h.
inline void check_gradient(const std::string& what, Tensor input,
                           const std::function<Tensor(Tensor&)>& loss_fn, float tol = 2e-2f) {
    input.set_requires_grad(true);
    input.zero_grad();

    Tensor loss = loss_fn(input);
    loss.backward();

    if (!input.has_grad()) {
        ++g_checks;
        ++g_failures;
        std::cout << "  [FAIL] " << what << " (no gradient propagated)\n";
        return;
    }

    const std::vector<float> analytic = input.grad().data();
    const float h = 1e-3f;
    float max_error = 0.0f;

    for (size_t i = 0; i < input.size(); ++i) {
        engine::autograd::NoGradGuard no_grad;  // the numeric evaluations need no graph
        const float original = input.data()[i];

        input.data()[i] = original + h;
        const float plus = loss_fn(input).data()[0];

        input.data()[i] = original - h;
        const float minus = loss_fn(input).data()[0];

        input.data()[i] = original;

        const float numeric = (plus - minus) / (2.0f * h);
        max_error = std::max(max_error, std::fabs(numeric - analytic[i]));
    }

    ++g_checks;
    if (max_error <= tol) {
        std::cout << "  [ ok ] " << what << " (max error " << std::scientific
                  << std::setprecision(2) << max_error << std::defaultfloat << ")\n";
    } else {
        ++g_failures;
        std::cout << "  [FAIL] " << what << " (max error " << max_error << " > " << tol << ")\n";
    }
}

}  // namespace testing

// Cada fichero de prueba expone su bloque; test_main.cpp los encadena.
void run_tensor_tests();
void run_autograd_tests();
void run_nn_tests();
void run_conv_tests();
void run_transformer_tests();
void run_reference_tests();
void run_cuda_parity_tests();
void run_cuda_indexing_tests();

#endif  // ENGINE_TEST_SUPPORT_HPP
