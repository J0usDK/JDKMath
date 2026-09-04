cmake_minimum_required(VERSION 3.14)
project(JDKMath VERSION 1.0.0 LANGUAGES CXX)

add_library(JDKMath INTERFACE)
add_library(JDK::Math ALIAS JDKMath)

target_include_directories(JDKMath INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_compile_features(JDKMath INTERFACE cxx_std_17)