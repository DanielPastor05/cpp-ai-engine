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
)

# Librería del motor, compartida por los ejemplos y las pruebas
add_library(engine STATIC ${SOURCES})

if(MSVC)
    target_compile_options(engine PRIVATE /W4)
else()
    target_compile_options(engine PRIVATE -Wall -Wextra)
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
    tests/test_reference.cpp)
target_include_directories(test_engine PRIVATE tests)
# Los ficheros de referencia se leen del arbol de fuentes, no del de compilacion
target_compile_definitions(test_engine PRIVATE
    ENGINE_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
target_link_libraries(test_engine PRIVATE engine)
add_test(NAME engine_tests COMMAND test_engine)

# ---------------------------------------------------------
# Instalación: permite consumir la librería desde otro proyecto con
#   find_package(cpp_ai_engine) ; target_link_libraries(app PRIVATE engine::engine)
# ---------------------------------------------------------
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

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
