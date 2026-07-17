# Speak — CMake build for Windows (MSVC). macOS/Linux use plugin/Makefile.
#
#   cmake -B build -A x64 -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake
#   cmake --build build --config Release
#
# Produces build/Release/Speak.ofx (OpenCL render path; CUDA is opt-in via
# -DSPEAK_ENABLE_CUDA=ON once the CUDA port has been verified on hardware).

cmake_minimum_required(VERSION 3.20)
project(Speak LANGUAGES CXX)

option(SPEAK_ENABLE_CUDA "Build the (hardware-unverified) CUDA render path" OFF)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCL REQUIRED)

set(SUPPORT ofx/Support/Library)
add_library(Speak MODULE
    plugin/SpeakOfxMain.cpp
    plugin/SpeakPlugin.cpp
    plugin/SpeakOpenCLKernel.cpp
    ${SUPPORT}/ofxsCore.cpp
    ${SUPPORT}/ofxsImageEffect.cpp
    ${SUPPORT}/ofxsInteract.cpp
    ${SUPPORT}/ofxsLog.cpp
    ${SUPPORT}/ofxsMultiThread.cpp
    ${SUPPORT}/ofxsParams.cpp
    ${SUPPORT}/ofxsProperty.cpp
    ${SUPPORT}/ofxsPropertyValidation.cpp
)

target_include_directories(Speak PRIVATE plugin ofx/include ofx/Support/include)
target_link_libraries(Speak PRIVATE OpenCL::OpenCL)
set_target_properties(Speak PROPERTIES PREFIX "" SUFFIX ".ofx")

if (MSVC)
    target_compile_definitions(Speak PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS)
    target_compile_options(Speak PRIVATE /O2 /EHsc)
endif()

if (SPEAK_ENABLE_CUDA)
    enable_language(CUDA)
    target_sources(Speak PRIVATE plugin/SpeakCudaKernel.cu)
    target_compile_definitions(Speak PRIVATE SPEAK_ENABLE_CUDA)
endif()
