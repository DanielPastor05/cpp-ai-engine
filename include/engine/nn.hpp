#ifndef ENGINE_NN_HPP
#define ENGINE_NN_HPP

#include "engine/tensor.hpp"

#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace engine {
namespace nn {

// ---------------------------------------------------------
// The base abstraction behind every layer
// ---------------------------------------------------------
class Module {
public:
    virtual ~Module() = default;

    virtual Tensor forward(const Tensor& input) = 0;

    // Syntactic sugar: module(x) is the same as module.forward(x)
    Tensor operator()(const Tensor& input) { return forward(input); }

    // The layer's trainable parameters (empty for stateless layers).
    // The returned Tensors share the internal implementation, so writing to
    // them updates the layer's weights.
    virtual std::vector<Tensor> parameters() { return {}; }

    // Parameters under a stable name, so a checkpoint can be saved and
    // reloaded matching each tensor with its own. The order must always agree
    // with parameters().
    virtual std::vector<std::pair<std::string, Tensor>> named_parameters(
        const std::string& prefix = "");

    // Training mode. Layers like Dropout behave differently when training and
    // when inferring; the switch propagates to sub-layers.
    virtual void train(bool mode = true) { training_ = mode; }
    void eval() { train(false); }
    bool is_training() const { return training_; }

    virtual std::string name() const { return "Module"; }

    // Zeroes the accumulated gradients of every parameter
    void zero_grad();

    // Total number of trainable scalars
    size_t num_parameters();

protected:
    bool training_ = true;
};

// ---------------------------------------------------------
// Dense (fully connected) layer: y = x . W + b
// ---------------------------------------------------------
class Linear : public Module {
public:
    // Uniform Xavier/Glorot initialisation: U(-limit, limit) with
    // limit = sqrt(6 / (in_features + out_features)).
    Linear(size_t in_features, size_t out_features, bool use_bias = true);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::string name() const override;

    Tensor& weight() { return weight_; }
    Tensor& bias() { return bias_; }
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }

private:
    size_t in_features_;
    size_t out_features_;
    bool use_bias_;
    Tensor weight_;  // (in_features, out_features)
    Tensor bias_;    // (1, out_features)
};

// ---------------------------------------------------------
// Activations (no parameters)
// ---------------------------------------------------------
class ReLU : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "ReLU"; }
};

class Softmax : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "Softmax"; }
};

class Sigmoid : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "Sigmoid"; }
};

class Tanh : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "Tanh"; }
};

// GELU in its hyperbolic-tangent approximation, the one GPT and BERT use.
// Unlike ReLU it has no kink at the origin, and it lets a little negative
// signal through instead of clamping it away entirely.
class GELU : public Module {
public:
    Tensor forward(const Tensor& input) override;
    std::string name() const override { return "GELU"; }
};

// Zeroes each activation with probability p during training and scales the
// rest by 1/(1-p), so the mean is preserved. In eval() mode it does nothing.
// It is regularisation: it forces the network not to depend on any one
// neuron.
class Dropout : public Module {
public:
    explicit Dropout(float p = 0.5f);

    Tensor forward(const Tensor& input) override;
    std::string name() const override;
    float probability() const { return p_; }

private:
    float p_;
};

// ---------------------------------------------------------
// Sequential container of layers
// ---------------------------------------------------------
class Sequential : public Module {
public:
    Sequential() = default;
    Sequential(std::initializer_list<std::shared_ptr<Module>> layers);

    Sequential& add(std::shared_ptr<Module> layer);

    Tensor forward(const Tensor& input) override;
    std::vector<Tensor> parameters() override;
    std::vector<std::pair<std::string, Tensor>> named_parameters(
        const std::string& prefix = "") override;
    void train(bool mode = true) override;
    std::string name() const override { return "Sequential"; }

    void summary() const;
    size_t size() const { return layers_.size(); }
    Module& at(size_t index);

private:
    std::vector<std::shared_ptr<Module>> layers_;
};

// Construction helpers
template <typename T, typename... Args>
std::shared_ptr<T> make(Args&&... args) {
    return std::make_shared<T>(std::forward<Args>(args)...);
}

// ---------------------------------------------------------
// Loss functions
// ---------------------------------------------------------

// Mean squared error: mean((pred - target)^2)
Tensor mse_loss(const Tensor& prediction, const Tensor& target);

// Cross entropy over unnormalised logits, with the correct classes given as
// integer indices. It fuses log-softmax and NLL into a single graph node: that
// is numerically more stable than chaining Softmax + log, and its gradient
// reduces to (softmax(logits) - one_hot) / N.
//
//   logits  : (N, C), unnormalised
//   targets : N class indices in [0, C)
Tensor cross_entropy_loss(const Tensor& logits, const std::vector<size_t>& targets);

class CrossEntropyLoss {
public:
    Tensor forward(const Tensor& logits, const std::vector<size_t>& targets) const {
        return cross_entropy_loss(logits, targets);
    }
    Tensor operator()(const Tensor& logits, const std::vector<size_t>& targets) const {
        return forward(logits, targets);
    }
};

// Index of the highest-scoring class per row (used to compute accuracy)
std::vector<size_t> argmax_rows(const Tensor& logits);
float accuracy(const Tensor& logits, const std::vector<size_t>& targets);

}  // namespace nn
}  // namespace engine

#endif  // ENGINE_NN_HPP
