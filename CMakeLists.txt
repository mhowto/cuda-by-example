#  Ref: https://devblogs.nvidia.com/building-cuda-applications-cmake/
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(cuda_by_example LANGUAGES CXX CUDA)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)

include_directories( ${OPENGL_INCLUDE_DIRS}  ${GLUT_INCLUDE_DIRS} )

add_executable(4-vecsum
        4-vecsum.cu
        common/book.h)
#set_property(TARGET 4-vecsum PROPERTY )

add_executable(4-mandel
        4-mandel.cu
        common/book.h
        common/cpu_bitmap.h)
target_link_libraries(4-mandel
        ${OPENGL_LIBRARIES} ${GLUT_LIBRARY})

add_executable(5-vecsum
        5-vecsum.cu
        common/book.h
        common/index.h
        common/index.cu)
set_property(TARGET 5-vecsum
        PROPERTY CUDA_SEPARABLE_COMPILATION ON)
