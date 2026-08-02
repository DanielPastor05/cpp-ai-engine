cmake_minimum_required(VERSION 3.18)
project(cpp_ai_engine VERSION 0.5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# The public headers
include_directories(include)

# The library's translation units
set(SOURCES
    src/tensor.cpp
    src/autograd.cpp
    src/nn.cpp
    src/conv.cpp
    src/transformer.cpp
    src/optim.cpp
    src/serialize.cpp
    src/data.cpp
    src/parallel.cpp
    src/storage.cpp
    src/cuda_disabled.cpp
)

# ---------------------------------------------------------
# The CUDA backend.
#
# Off by default on purpose. The engine has to keep building and keep passing
# its tests on a machine with neither toolkit nor GPU, which is what CI runs on;
# the backend is an optional acceleration, not a requirement.
#
#   cmake -B build-cuda -S . -DENGINE_CUDA=ON
# ---------------------------------------------------------
option(ENGINE_CUDA "Build the CUDA backend" OFF)

if(ENGINE_CUDA)
    # The default has to be set BEFORE enable_language(CUDA).
    #
    # Since CMake 3.18, enable_language(CUDA) already initialises
    # CMAKE_CUDA_ARCHITECTURES from the compiler's own default, so an
    # if(NOT DEFINED ...) placed after it is never true and the value here is
    # ignored in silence. The symptom is not a build error: the binary comes out
    # with cubins for another architecture, the driver cannot use them, it falls
    # back to compiling the PTX at run time, and fails there with "the provided
    # PTX was compiled with an unsupported toolchain".
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 86 is the compute capability of Ampere GA104 (RTX 30xx), which is the
        # machine this was developed on. Override with
        # -DCMAKE_CUDA_ARCHITECTURES=<xx>, or list several separated by
        # semicolons to produce a binary that works on more cards.
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()

    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    list(APPEND SOURCES src/cuda/runtime.cu src/cuda/kernels.cu)
    message(STATUS "CUDA backend enabled (architectures: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

# The engine library, shared by the examples and the tests
find_package(Threads REQUIRED)
add_library(engine STATIC ${SOURCES})

if(ENGINE_CUDA)
    # PUBLIC and not PRIVATE, because the macro changes Storage's layout: with
    # CUDA it carries a device pointer and two validity flags. If the library
    # and whoever uses it disagreed about that, the size mismatch would give
    # memory corruption rather than a compile error.
    target_compile_definitions(engine PUBLIC ENGINE_CUDA)
    # Static and not Shared: nvcc 13 deprecates the '-cudart shared' that CMake
    # generates for Shared, and warns once per compiled file. CMake does not
    # expose the 'hybrid' the warning suggests, so linking the static runtime is
    # what gives clean output. It also stops the binary depending on the right
    # cudart being on PATH.
    set_target_properties(engine PROPERTIES CUDA_RUNTIME_LIBRARY Static)
endif()

# The warnings are for the host compiler: nvcc does not understand these flags.
if(MSVC)
    target_compile_options(engine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(engine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)
endif()

# Tensors
add_executable(cpp_ai_engine examples/main.cpp)
target_link_libraries(cpp_ai_engine PRIVATE engine)

# Autograd
add_executable(autograd_demo examples/autograd_demo.cpp)
target_link_libraries(autograd_demo PRIVATE engine)

# Layers and optimisers
add_executable(nn_demo examples/nn_demo.cpp)
target_link_libraries(nn_demo PRIVATE engine)

# Convolutional networks
add_executable(cnn_demo examples/cnn_demo.cpp)
target_link_libraries(cnn_demo PRIVATE engine)

# Transformer
add_executable(transformer_demo examples/transformer_demo.cpp)
target_link_libraries(transformer_demo PRIVATE engine)

# Real data
add_executable(mnist_demo examples/mnist_demo.cpp)
target_link_libraries(mnist_demo PRIVATE engine)

# The performance benchmark. Not run in CI: it measures, it does not verify.
add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE engine)

# The matrix-product benchmark, kept separate so it can be handed to a profiler:
# one kernel, one shape, nothing else around it to measure.
add_executable(bench_matmul bench/bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE engine)

# cuBLAS is linked HERE ONLY, and only as the benchmark's reference row. The
# engine never calls it: the point of the project is to write the kernel. But
# measuring against NVIDIA's own library is the only way the number means
# anything, and not measuring reads as dodging the comparison.
#
# If the toolkit does not ship cuBLAS the row simply does not appear and the
# rest of the benchmark works unchanged.
if(ENGINE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(TARGET CUDA::cublas)
        target_link_libraries(bench_matmul PRIVATE CUDA::cublas)
        target_compile_definitions(bench_matmul PRIVATE ENGINE_BENCH_CUBLAS)
        message(STATUS "bench_matmul: cuBLAS reference row enabled")
    else()
        message(STATUS "bench_matmul: cuBLAS not found; its row is skipped")
    endif()
endif()

# The test suite
enable_testing()
add_executable(test_engine
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_autograd.cpp
    tests/test_nn.cpp
    tests/test_conv.cpp
    tests/test_transformer.cpp
    tests/test_reference.cpp
    tests/test_cuda_indexing.cpp
    tests/test_cuda_parity.cpp)
target_include_directories(test_engine PRIVATE tests)
# The fixtures are read from the source tree, not the build tree
target_compile_definitions(test_engine PRIVATE
    ENGINE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    ENGINE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_engine PRIVATE engine)
add_test(NAME engine_tests COMMAND test_engine)

# ---------------------------------------------------------
# Installation: lets another project consume the library with
#   find_package(cpp_ai_engine) ; target_link_libraries(app PRIVATE engine::engine)
# ---------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

target_link_libraries(engine PUBLIC Threads::Threads)
target_include_directories(engine PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

install(TARGETS engine EXPORT cpp_ai_engineTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT cpp_ai_engineTargets
        FILE cpp_ai_engineTargets.cmake
        NAMESPACE engine::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpp_ai_engine)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/cpp_ai_engineConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cpp_ai_engineConfig.cmake"
     "include(\"\${CMAKE_CURRENT_LIST_DIR}/cpp_ai_engineTargets.cmake\")\n")

install(FILES
        "${CMAKE_CURRENT_BINARY_DIR}/cpp_ai_engineConfig.cmake"
        "${CMAKE_CURRENT_BINARY_DIR}/cpp_ai_engineConfigVersion.cmake"
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpp_ai_engine)
