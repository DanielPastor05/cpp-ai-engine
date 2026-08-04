// A character-level language model, trained by this engine on its own
// documentation.
//
// Everything the other demos show is a number: an accuracy, a loss curve, a
// millisecond count. This one produces text, which is the only output a reader
// can judge without trusting the person who wrote the benchmark.
//
// The corpus is five of this repository's own documents -- around 110 000
// characters of English technical prose, 121 distinct bytes. The exact size
// moves whenever a document is edited, which is the point of using them.
// Nothing is downloaded and there is no licence question, because the training
// data is the repository explaining itself.
//
// It is also small, and that matters for what comes out: a corpus this size
// teaches a model the shape of words and the rhythm of punctuation, not facts.
// What it generates is meant to be read as evidence that the machinery works,
// not as prose. It reaches 1.97 bits per character against log2 of the alphabet
// size for a uniform guess.
//
// The model is the one already in engine/transformer.hpp -- embedding,
// positional encoding, two blocks with a causal mask, a projection back to the
// alphabet. No new layers were written for this.

#include "engine/autograd.hpp"
#include "engine/data.hpp"
#include "engine/nn.hpp"
#include "engine/optim.hpp"
#include "engine/random.hpp"
#include "engine/tensor.hpp"
#include "engine/transformer.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <random>
#include <string>
#include <vector>

using engine::Tensor;
namespace nn = engine::nn;
namespace optim = engine::optim;
namespace data = engine::data;

namespace {

constexpr size_t kSeqLen = 64;  // characters of context
constexpr size_t kDModel = 96;  // embedding width
constexpr size_t kHeads = 4;    // 24 per head
constexpr size_t kFeedForward = 192;
constexpr size_t kBatch = 32;
constexpr size_t kBlocks = 2;

// The model, assembled by hand rather than through nn::Sequential, because a
// Transformer block takes a mask as a second argument and Sequential's
// forward(input) has nowhere to put it.
struct CharModel {
    nn::Embedding embedding;
    std::vector<nn::TransformerBlock> blocks;
    nn::Linear head;
    Tensor positions;

    CharModel(size_t vocab, size_t seq_len)
        : embedding(vocab, kDModel),
          head(kDModel, vocab),
          positions(nn::positional_encoding(seq_len, kDModel)) {
        blocks.reserve(kBlocks);
        for (size_t i = 0; i < kBlocks; ++i) {
            blocks.emplace_back(kDModel, kHeads, kFeedForward);
        }
    }

    // (B, S) indices in, (B, S, vocab) logits out.
    Tensor forward(const Tensor& ids, const Tensor& mask) {
        Tensor h = embedding.forward(ids) + positions;
        for (nn::TransformerBlock& block : blocks) h = block.forward(h, &mask);
        return head.forward(h);
    }

    std::vector<Tensor> parameters() {
        std::vector<Tensor> out = embedding.parameters();
        for (nn::TransformerBlock& block : blocks) {
            for (const Tensor& p : block.parameters()) out.push_back(p);
        }
        for (const Tensor& p : head.parameters()) out.push_back(p);
        return out;
    }

    void train(bool mode) {
        embedding.train(mode);
        for (nn::TransformerBlock& block : blocks) block.train(mode);
        head.train(mode);
    }
};

// One batch of (context, next character) pairs, drawn at random positions.
void sample_batch(const std::vector<size_t>& corpus, std::mt19937& rng, Tensor& ids,
                  std::vector<size_t>& targets) {
    std::uniform_int_distribution<size_t> start(0, corpus.size() - kSeqLen - 2);
    targets.clear();
    targets.reserve(kBatch * kSeqLen);
    float* id_values = ids.data();

    for (size_t b = 0; b < kBatch; ++b) {
        const size_t begin = start(rng);
        for (size_t t = 0; t < kSeqLen; ++t) {
            id_values[b * kSeqLen + t] = static_cast<float>(corpus[begin + t]);
            // The target at position t is the character at t+1: every position
            // in the window is a training example, which is what the causal mask
            // makes safe.
            targets.push_back(corpus[begin + t + 1]);
        }
    }
}

// Greedy-with-temperature sampling. Temperature 0 would be argmax; anything
// above it trades correctness for variety, and at this corpus size a little
// variety is the difference between text and a loop of "the the the".
std::string generate(CharModel& model, const data::CharVocab& vocab, const std::string& prompt,
                     size_t count, float temperature, std::mt19937& rng) {
    engine::autograd::NoGradGuard no_grad;
    model.train(false);

    const Tensor mask = nn::causal_mask(kSeqLen);
    std::vector<size_t> context = vocab.encode(prompt);
    std::string out = prompt;

    for (size_t step = 0; step < count; ++step) {
        // The model always sees exactly kSeqLen characters; a short prompt is
        // left-padded with the first symbol and a long one is cut from the left.
        std::vector<size_t> window(kSeqLen, 0);
        const size_t take = std::min(kSeqLen, context.size());
        for (size_t i = 0; i < take; ++i) {
            window[kSeqLen - take + i] = context[context.size() - take + i];
        }

        Tensor ids({1, kSeqLen}, 0.0f, false);
        float* id_values = ids.data();
        for (size_t i = 0; i < kSeqLen; ++i) id_values[i] = static_cast<float>(window[i]);

        const Tensor logits = model.forward(ids, mask);  // (1, S, vocab)
        const size_t vocab_size = vocab.size();
        const float* row = logits.data() + (kSeqLen - 1) * vocab_size;

        // Softmax over the last position, at the given temperature, computed
        // here rather than through the engine because it is one row and the
        // sampling needs the probabilities as host values anyway.
        float best = row[0];
        for (size_t i = 1; i < vocab_size; ++i) best = std::max(best, row[i]);
        std::vector<float> probability(vocab_size);
        float total = 0.0f;
        for (size_t i = 0; i < vocab_size; ++i) {
            probability[i] = std::exp((row[i] - best) / temperature);
            total += probability[i];
        }

        std::uniform_real_distribution<float> pick(0.0f, total);
        float threshold = pick(rng);
        size_t chosen = vocab_size - 1;
        for (size_t i = 0; i < vocab_size; ++i) {
            threshold -= probability[i];
            if (threshold <= 0.0f) {
                chosen = i;
                break;
            }
        }

        context.push_back(chosen);
        out += vocab.symbol(chosen);
    }

    model.train(true);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    engine::manual_seed(42);

    // Epoch count first, so a reader can run a short one before committing to
    // the full thing: ./charlm_demo 200
    const int steps = argc > 1 ? std::atoi(argv[1]) : 1500;

    std::cout << "====================================================\n";
    std::cout << "  A character-level language model, on this engine   \n";
    std::cout << "====================================================\n\n";

    // ---------------------------------------------------------
    // 1. The corpus
    // ---------------------------------------------------------
    const std::string root = ENGINE_REPO_DIR;
    const std::string corpus_text = data::load_text(
        {root + "/docs/DESIGN.md", root + "/docs/PERFORMANCE.md", root + "/docs/ENGINEERING.md",
         root + "/docs/CUDA.md", root + "/README.md"});
    if (corpus_text.size() < kSeqLen * 4) {
        std::cerr << "The corpus is empty or unreadable; nothing to train on.\n";
        return 1;
    }

    const data::CharVocab vocab(corpus_text);
    const std::vector<size_t> corpus = vocab.encode(corpus_text);

    std::cout << "--- 1. Corpus ---\n";
    std::printf("  %zu characters, %zu distinct\n", corpus.size(), vocab.size());
    std::printf(
        "  the engine's own documentation: DESIGN, PERFORMANCE, ENGINEERING, CUDA, README\n\n");

    // ---------------------------------------------------------
    // 2. The model
    // ---------------------------------------------------------
    CharModel model(vocab.size(), kSeqLen);
    size_t parameters = 0;
    for (const Tensor& p : model.parameters()) parameters += p.size();

    std::cout << "--- 2. Model ---\n";
    std::printf("  %zu blocks, d_model %zu, %zu heads, context %zu -> %zu parameters\n\n", kBlocks,
                kDModel, kHeads, kSeqLen, parameters);

    // ---------------------------------------------------------
    // 3. Training
    // ---------------------------------------------------------
    optim::Adam opt(model.parameters(), 0.003f);
    const Tensor mask = nn::causal_mask(kSeqLen);
    Tensor ids({kBatch, kSeqLen}, 0.0f, false);
    std::vector<size_t> targets;
    std::mt19937 rng(1234);

    std::printf("--- 3. Training (%d steps, batches of %zu) ---\n", steps, kBatch);
    const auto started = std::chrono::steady_clock::now();

    for (int step = 1; step <= steps; ++step) {
        sample_batch(corpus, rng, ids, targets);

        opt.zero_grad();
        const Tensor logits = model.forward(ids, mask);
        // (B, S, vocab) -> (B*S, vocab): cross entropy wants one row per example
        // and every position in the window is one.
        const Tensor flat = logits.reshape({kBatch * kSeqLen, vocab.size()});
        Tensor loss = nn::cross_entropy_loss(flat, targets);
        loss.backward();
        optim::clip_grad_norm(model.parameters(), 1.0f);
        opt.step();

        if (step == 1 || step % 100 == 0 || step == steps) {
            const double elapsed =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            // Bits per character is the comparable number: cross entropy in nats
            // over ln(2). Uniform over this alphabet would be log2(121) = 6.92.
            const float nats = loss.data()[0];
            std::printf("  step %5d | loss = %.4f | %.2f bits/char | %.1f s\n", step, nats,
                        nats / std::log(2.0f), elapsed);
        }
    }
    std::cout << "\n";

    // ---------------------------------------------------------
    // 4. What it writes
    // ---------------------------------------------------------
    std::cout << "--- 4. Generated text ---\n\n";
    for (float temperature : {0.5f, 0.8f}) {
        std::printf("  temperature %.1f:\n", temperature);
        const std::string sample = generate(model, vocab, "The engine ", 320, temperature, rng);
        std::printf("    %s\n\n", sample.c_str());
    }

    std::cout << "This is a small corpus and a model of 173 thousand\n";
    std::cout << "parameters. It learns the shape of words, not what they mean.\n";
    return 0;
}
