// libFuzzer target for the weight-file parser.
//
// This is the only input to the engine that does not come from its own API. A
// checkpoint is a thing people download, and `src/serialize.cpp` reads three
// 32-bit sizes out of its header and allocates on all of them.
//
// Reading that loop with a fuzzer in mind found a bug before the fuzzer ran: the
// element count was a product of 64-bit dimensions accumulated into a size_t
// with no overflow check, so two dimensions of 2^32 wrapped to zero and the
// tensor loaded empty while its shape claimed 2^64 elements. Nothing crashed and
// nothing was reported. tests/test_nn.cpp pins that case; this keeps looking for
// the ones nobody thought of.
//
// The parser is expected to **throw** on anything malformed, never to read out
// of bounds and never to allocate on an unvalidated size. So every exception is
// caught and treated as success -- a rejected file is the correct outcome. What
// the fuzzer is looking for is the other kind of failure: a sanitiser report, a
// crash, or a hang.
//
// Build and run (needs clang; MSVC has no libFuzzer):
//
//   clang++ -std=c++17 -g -O1 -fsanitize=fuzzer,address,undefined \
//       -Iinclude tests/fuzz_serialize.cpp src/*.cpp -o fuzz_serialize
//   ./fuzz_serialize -max_total_time=60 corpus/
//
// It writes the input to a temporary file because the public API takes a path
// rather than a buffer. That costs a syscall per case and keeps the fuzzer
// pointed at the parser the engine actually ships, rather than at a copy of it
// written to be fuzzable.

#include "engine/serialize.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, size_t size) {
    // A cap, because the interesting inputs are headers rather than payloads and
    // an enormous case only slows the campaign down.
    if (size > 64 * 1024) return 0;

    const std::string path = "fuzz_case.bin";
    {
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) return 0;
        out.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size));
    }

    // inspect_parameters is the whole parse without the model-matching that
    // follows it: it is the part reading attacker-controlled bytes.
    try {
        (void)engine::inspect_parameters(path);
    } catch (const std::exception&) {
        // The documented outcome for a malformed file. Not a finding.
    }

    try {
        (void)engine::load_tensors(path);
    } catch (const std::exception&) {
    }

    std::remove(path.c_str());
    return 0;
}
