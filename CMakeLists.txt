cmake_minimum_required(VERSION 3.18)
project(cpp_ai_engine VERSION 0.5 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Tipo de compilacion" FORCE)
endif()

# Incluir la carpeta de cabeceras (headers)
include_directories(include)

# Archivos de código fuente de la librería
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
# Fase 6: backend CUDA.
#
# Está apagado por defecto a propósito. El motor tiene que seguir compilando y
# pasando las pruebas en una máquina sin toolkit ni GPU, que es lo que hay en
# CI; el backend es una aceleración opcional, no un requisito.
#
#   cmake -B build-cuda -S . -DENGINE_CUDA=ON
# ---------------------------------------------------------
option(ENGINE_CUDA "Compilar el backend CUDA (Fase 6)" OFF)

if(ENGINE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # 86 es la capacidad de cómputo de Ampere GA104 (RTX 30xx), que es la
        # máquina donde se desarrolló. Se puede cambiar con
        # -DCMAKE_CUDA_ARCHITECTURES=<xx>, o poner varias separadas por ';'
        # para generar un binario que valga en más tarjetas.
        set(CMAKE_CUDA_ARCHITECTURES 86)
    endif()
    list(APPEND SOURCES src/cuda/runtime.cu src/cuda/kernels.cu)
    message(STATUS "Backend CUDA activado (arquitecturas: ${CMAKE_CUDA_ARCHITECTURES})")
endif()

# Librería del motor, compartida por los ejemplos y las pruebas
find_package(Threads REQUIRED)
add_library(engine STATIC ${SOURCES})

if(ENGINE_CUDA)
    # PUBLIC, y no PRIVATE, porque la macro cambia la disposición de Storage:
    # con CUDA lleva puntero de dispositivo y banderas de validez. Si la
    # librería y quien la usa no coincidieran en esto, el desajuste de tamaño
    # daría corrupción de memoria en lugar de un error de compilación.
    target_compile_definitions(engine PUBLIC ENGINE_CUDA)
    set_target_properties(engine PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
endif()

# Los avisos son para el compilador de host: nvcc no entiende estas banderas.
if(MSVC)
    target_compile_options(engine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/W4>)
else()
    target_compile_options(engine PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra>)
endif()

# Ejecutable de la Fase 1 (Tensores)
add_executable(cpp_ai_engine examples/main.cpp)
target_link_libraries(cpp_ai_engine PRIVATE engine)

# Ejecutable de la Fase 2 (Autograd)
add_executable(autograd_demo examples/autograd_demo.cpp)
target_link_libraries(autograd_demo PRIVATE engine)

# Ejecutable de la Fase 3 (Capas y Optimizadores)
add_executable(nn_demo examples/nn_demo.cpp)
target_link_libraries(nn_demo PRIVATE engine)

# Ejecutable de la Fase 4 (Redes Convolucionales)
add_executable(cnn_demo examples/cnn_demo.cpp)
target_link_libraries(cnn_demo PRIVATE engine)

# Ejecutable de la Fase 5 (Transformer)
add_executable(transformer_demo examples/transformer_demo.cpp)
target_link_libraries(transformer_demo PRIVATE engine)

# Ejemplo con datos reales (MNIST)
add_executable(mnist_demo examples/mnist_demo.cpp)
target_link_libraries(mnist_demo PRIVATE engine)

# Banco de pruebas de rendimiento (no se ejecuta en CI: mide, no verifica)
add_executable(bench bench/bench.cpp)
target_link_libraries(bench PRIVATE engine)

# Suite de pruebas
enable_testing()
add_executable(test_engine
    tests/test_main.cpp
    tests/test_tensor.cpp
    tests/test_autograd.cpp
    tests/test_nn.cpp
    tests/test_conv.cpp
    tests/test_transformer.cpp
    tests/test_reference.cpp
    tests/test_cuda_parity.cpp)
target_include_directories(test_engine PRIVATE tests)
# Los ficheros de referencia se leen del arbol de fuentes, no del de compilacion
target_compile_definitions(test_engine PRIVATE
    ENGINE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures"
    ENGINE_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")
target_link_libraries(test_engine PRIVATE engine)
add_test(NAME engine_tests COMMAND test_engine)

# ---------------------------------------------------------
# Instalación: permite consumir la librería desde otro proyecto con
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
