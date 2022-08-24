cmake_minimum_required(VERSION 3.0.0)
project(My_Gemm VERSION 0.1.0)

# set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -mavx")
set(BLA_VENDOR FLAME)
find_package(BLAS)

if(NOT BLAS_FOUND)
	message(SEND_ERROR "Failed to find BLAS")
	return()
else()
	include_directories(${BLAS_INCLUDE_DIR})
endif()

include_directories(include)
set(SOURCE src/main.cpp
           src/stuff_for_matrix.cpp
           src/gemm.cpp)

add_executable(My_Gemm ${SOURCE})
target_link_libraries(My_Gemm ${BLAS_LIBRARIES})
