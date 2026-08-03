// The smallest program that proves an installed cpp-ai-engine is usable.
//
// It is short on purpose. What is under test is not the engine -- 530 checks
// already cover that -- but the four things between a consumer and the engine,
// each of which fails differently:
//
//   configure  the exported package resolves, with its dependencies
//   compile    the installed headers are self-contained and on the include path
//   link       the installed archive has the symbols the headers promised
//   run        the layout the consumer compiled against matches the library's
//
// The last one is the reason this executes rather than just building. With a
// CUDA build ENGINE_CUDA is a PUBLIC compile definition because it changes
// Storage's layout; if it ever stopped travelling with the exported target, a
// consumer would compile and link cleanly and then corrupt memory. A run that
// checks a number catches that. A build that only compiles does not.

#include "engine/autograd.hpp"
#include "engine/nn.hpp"
#include "engine/tensor.hpp"

#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    // Autograd, because it is the part that reaches furthest into the library:
    // a graph, a backward pass and the storage underneath all three.
    engine::Tensor x({1}, 3.0f, true);
    engine::Tensor y = x * x;  // dy/dx = 2x = 6
    y.backward();

    const float gradient = x.grad().data()[0];
    std::printf("d(x^2)/dx at x=3 -> %.1f (expected 6.0)\n", gradient);

    // A layer, because nn/ is a separate translation unit in the archive and a
    // partial install would show up here and nowhere else.
    engine::nn::Linear layer(4, 2);
    const engine::Tensor out = layer.forward(engine::Tensor({3, 4}, 1.0f, false));
    std::printf("Linear(4,2) on a (3,4) batch -> %s (expected (3, 2))\n", out.shape_str().c_str());

    const bool ok = std::fabs(gradient - 6.0f) < 1e-6f && out.shape() == std::vector<size_t>{3, 2};

    std::printf("%s\n", ok ? "package test PASSED" : "package test FAILED");
    return ok ? 0 : 1;
}
