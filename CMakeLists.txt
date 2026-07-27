cmake_minimum_required(VERSION 3.18)
project(cpp_ai_engine VERSION 0.3 LANGUAGES CXX)

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
    src/optim.cpp
)

# Librería del motor, compartida por los ejemplos y las pruebas
add_library(engine STATIC ${SOURCES})
target_include_directories(engine PUBLIC include)

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

# Suite de pruebas
enable_testing()
add_executable(test_engine tests/test_engine.cpp)
target_link_libraries(test_engine PRIVATE engine)
add_test(NAME engine_tests COMMAND test_engine)
