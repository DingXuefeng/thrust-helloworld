cmake_minimum_required(VERSION 3.15)
project(Thrust_HelloWorld LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

#set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -lineinfo --ptxas-options=-v -keep --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo --extended-lambda")
set(CMAKE_CUDA_ARCHITECTURES "86") # Amphere for 3060 Ti

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust HOST CPP DEVICE CUDA)

add_executable(Thrust_HelloWorld main.cu)
target_link_libraries(Thrust_HelloWorld PRIVATE Thrust::Thrust nvToolsExt) 
# Pass the USE_FLOAT definition to the compiler if it's set
if(USE_FLOAT)
  target_compile_definitions(Thrust_HelloWorld PRIVATE USE_FLOAT)
endif()

add_executable(get_compute_capability get_compute_capability.cu)

add_executable(naive naive.cu)
target_link_libraries(naive PRIVATE Thrust::Thrust)

# both static: ok
add_library(libB STATIC libB.cu)
add_library(libA STATIC libA.cu)
target_link_libraries(libA PRIVATE libB Thrust::Thrust)

add_executable(separate separate.cxx)
target_link_libraries(separate PRIVATE libA)

# in 1 lib: ok
add_library(libgpu SHARED libA.cu libB.cu)
target_link_libraries(libgpu PRIVATE libB Thrust::Thrust)

add_executable(separate2 separate.cxx)
target_link_libraries(separate2 PRIVATE libgpu)

# lib: ok
add_library(libBs STATIC libB.cu)
set_target_properties(libBs PROPERTIES POSITION_INDEPENDENT_CODE ON)
add_library(libAs SHARED libA.cu)
target_link_libraries(libAs PRIVATE libBs Thrust::Thrust)

add_executable(separate3 separate.cxx)
target_link_libraries(separate3 PRIVATE libAs)

# separable compilation avoided: ok
add_library(libgpu2 STATIC all_gpu.cu)
target_link_libraries(libgpu2 PRIVATE Thrust::Thrust)
add_executable(separate4 separate.cxx)
target_link_libraries(separate4 PRIVATE libgpu2)
