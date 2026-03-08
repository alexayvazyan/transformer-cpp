cmake_minimum_required(VERSION 3.10)
project(MyProject)
find_package(Eigen3 REQUIRED)
find_package(ZLIB REQUIRED)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -O0 -g")

add_executable(main
    input.cpp
    matrixlin.cpp
    helpers.cpp
    xgboost.cpp
)
target_link_libraries(main Eigen3::Eigen)

add_executable(findsqrt
    findsqrt.cpp
)

add_executable(dataplay
    dataplay.cpp
)

target_link_libraries(dataplay ZLIB::ZLIB)

add_executable(transformer_init
    transformer_init.cpp
    transformer_weight_storage.cpp
    transformer_save.cpp
)

add_executable(transformer_train
    transformer_train.cpp
    transformer_load.cpp
    transformer_weight_storage.cpp
    transformer_save.cpp
    helpers.cpp
)
target_link_libraries(transformer_train Eigen3::Eigen)