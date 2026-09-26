cmake_minimum_required(VERSION 3.13)

find_package(Python3 REQUIRED)

# HLSL kernels, compiled at runtime to cs_5_0 with d3dcompiler_47.dll
set(SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/hlsl")
set(SHADER_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
set(SHADER_HEADER "${SHADER_OUTPUT_DIR}/ggml-d3d11-shaders.hpp")
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

file(GLOB HLSL_SHADER_FILES
    "${SHADER_DIR}/*.hlsl"
    "${SHADER_DIR}/*.hlsli"
)

add_custom_command(
    OUTPUT ${SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SHADER_OUTPUT_DIR}
    COMMAND ${CMAKE_COMMAND} -E env PYTHONIOENCODING=utf-8
        ${Python3_EXECUTABLE} ${SHADER_DIR}/embed_hlsl.py
            --input_dir "${SHADER_DIR}"
            --output_file "${SHADER_HEADER}"
            --table
    DEPENDS ${HLSL_SHADER_FILES} ${SHADER_DIR}/embed_hlsl.py
    VERBATIM
)

add_custom_target(generate_d3d11_shaders DEPENDS ${SHADER_HEADER})

ggml_add_backend_library(ggml-d3d11
    ggml-d3d11.cpp
    ${SHADER_HEADER}
    ../../include/ggml-d3d11.h
)

add_dependencies(ggml-d3d11 generate_d3d11_shaders)

target_compile_features(ggml-d3d11 PRIVATE cxx_std_17)
target_include_directories(ggml-d3d11 PRIVATE ${SHADER_OUTPUT_DIR})
target_link_libraries(ggml-d3d11 PRIVATE d3d11 dxgi dxguid d3dcompiler_47)

# no D3D12 backend in this tree to step aside for: D3D11 is on by default
target_compile_definitions(ggml-d3d11 PRIVATE GGML_D3D11_DEFAULT_ON)
