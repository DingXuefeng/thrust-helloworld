cmake_minimum_required(VERSION 3.15)
project(Thrust_HelloWorld LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -lineinfo --ptxas-options=-v -keep")
set(CMAKE_CUDA_ARCHITECTURES "70;86") # Amphere for 3060 Ti

find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust HOST CPP DEVICE CUDA)

add_executable(Thrust_HelloWorld main.cu)
target_link_libraries(Thrust_HelloWorld PRIVATE Thrust::Thrust nvToolsExt) 
# Pass the USE_FLOAT definition to the compiler if it's set
if(USE_FLOAT)
  target_compile_definitions(Thrust_HelloWorld PRIVATE USE_FLOAT)
endif()

add_executable(get_compute_capability get_compute_capability.cu)
