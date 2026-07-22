cmake_minimum_required(VERSION 3.18)
project(cpp_ai_engine VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Incluir la carpeta de cabeceras (headers)
include_directories(include)

# Archivos de código fuente de la librería
set(SOURCES
    src/tensor.cpp
)

# Crear el ejecutable principal para probar la librería
add_executable(cpp_ai_engine examples/main.cpp ${SOURCES})
