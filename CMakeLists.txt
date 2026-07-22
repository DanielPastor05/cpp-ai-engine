cmake_minimum_required(VERSION 3.18)
project(cpp_ai_engine VERSION 0.2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluir la carpeta de cabeceras (headers)
include_directories(include)

# Archivos de código fuente de la librería
set(SOURCES
    src/tensor.cpp
    src/autograd.cpp
)

# Ejecutable de la Fase 1
add_executable(cpp_ai_engine examples/main.cpp ${SOURCES})

# Ejecutable de demostración de la Fase 2 (Autograd)
add_executable(autograd_demo examples/autograd_demo.cpp ${SOURCES})
