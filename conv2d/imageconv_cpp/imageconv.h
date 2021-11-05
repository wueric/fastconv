#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

#include <stdlib.h>
#include <stdint.h>

#include <iostream>

#include "fast2dconv.h"

#if defined(ENABLE_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;

inline omp_int_t omp_get_thread_num() { return 0; }

inline omp_int_t omp_get_max_threads() { return 1; }

inline void omp_set_num_threads(int num_threads) { return; }

#endif

namespace py=pybind11;

template<class T>
py::array_t<T, py::array::c_style | py::array::forcecast> batch_smallfilter_2dconv_same(
        py::array_t<T, py::array::c_style | py::array::forcecast> batched_image_matrix,
        py::array_t<T, py::array::c_style | py::array::forcecast> kernel_matrix,
        T pad_val) {


    py::buffer_info kernel_info = kernel_matrix.request();
    T *kernel_matrix_ptr = static_cast<T *> (kernel_info.ptr);
    const int64_t kernel_height = kernel_info.shape[0];
    const int64_t kernel_width = kernel_info.shape[1];


    py::buffer_info batched_image_info = batched_image_matrix.request();
    T *image_matrix_ptr = static_cast<T *> (batched_image_info.ptr);
    const int64_t n_images = batched_image_info.shape[0];
    const int64_t height = batched_image_info.shape[1];
    const int64_t width = batched_image_info.shape[2];

    const int64_t offset_per_image = height * width;

    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(T),     /* Size of one item */
            py::format_descriptor<T>::value, /* Buffer format */
            3,          /* How many dimensions? */
            {n_images, height, width}, /* Number of elements for each dimension */
            {sizeof(T) * height * width, sizeof(T) * width, sizeof(T)}  /* Strides for each dimension */
    );

    py::array_t<T> convolved_output = py::array_t<T>(output_buffer_info);
    py::buffer_info output_info = convolved_output.request();
    T *output_data_ptr = static_cast<T *> (output_info.ptr);

    omp_set_num_threads(16); // Use 16 threads for all consecutive parallel regions
    #pragma omp parallel for
    for (int64_t i = 0; i < n_images; ++i) {

        T *read_offset = i * offset_per_image + image_matrix_ptr;
        T *write_offset = i * offset_per_image + output_data_ptr;

        TwoDContigArrayWrapper<T> input_wrapper = TwoDContigArrayWrapper<T>(
                read_offset, height, width);
        TwoDContigArrayWrapper<T> kernel_wrapper = TwoDContigArrayWrapper<T>(
                kernel_matrix_ptr, kernel_height, kernel_width);
        TwoDContigArrayWrapper<T> write_wrapper = TwoDContigArrayWrapper<T>(
                write_offset, height, width);

        conv2D_samesize(input_wrapper, kernel_wrapper, write_wrapper, pad_val);

    }

    return convolved_output;
}

template<class T>
py::array_t<T, py::array::c_style | py::array::forcecast> batch_smallfilter_2dconv_shrink(
        py::array_t<T, py::array::c_style | py::array::forcecast> batched_image_matrix,
        py::array_t<T, py::array::c_style | py::array::forcecast> kernel_matrix) {

    py::buffer_info kernel_info = kernel_matrix.request();
    T *kernel_matrix_ptr = static_cast<T *> (kernel_info.ptr);
    const int64_t kernel_height = kernel_info.shape[0];
    const int64_t kernel_width = kernel_info.shape[1];

    const int64_t kernel_half_height = kernel_height >> 1;
    const int64_t kernel_half_width = kernel_width >> 1;


    py::buffer_info batched_image_info = batched_image_matrix.request();
    T *image_matrix_ptr = static_cast<T *> (batched_image_info.ptr);
    const int64_t n_images = batched_image_info.shape[0];
    const int64_t input_height = batched_image_info.shape[1];
    const int64_t input_width = batched_image_info.shape[2];

    const int64_t height = input_height - 2 * kernel_half_height;
    const int64_t width = input_width - 2 * kernel_half_width;

    const int64_t read_offset_per_image = input_height * input_width;
    const int64_t offset_per_image = height * width;

    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(T),     /* Size of one item */
            py::format_descriptor<T>::value, /* Buffer format */
            3,          /* How many dimensions? */
            {n_images, height, width}, /* Number of elements for each dimension */
            {sizeof(T) * height * width, sizeof(T) * width, sizeof(T)}  /* Strides for each dimension */
    );

    py::array_t<T> convolved_output = py::array_t<T>(output_buffer_info);
    py::buffer_info output_info = convolved_output.request();
    T *output_data_ptr = static_cast<T *> (output_info.ptr);

    omp_set_num_threads(16); // Use 16 threads for all consecutive parallel regions
    #pragma omp parallel for
    for (int64_t i = 0; i < n_images; ++i) {

        T *read_offset = i * read_offset_per_image + image_matrix_ptr;
        T *write_offset = i * offset_per_image + output_data_ptr;

        TwoDContigArrayWrapper<T> input_wrapper = TwoDContigArrayWrapper<T>(
                read_offset, input_height, input_width);
        TwoDContigArrayWrapper<T> kernel_wrapper = TwoDContigArrayWrapper<T>(
                kernel_matrix_ptr, kernel_height, kernel_width);
        TwoDContigArrayWrapper<T> write_wrapper = TwoDContigArrayWrapper<T>(
                write_offset, height, width);

        conv2D_shrinksize(input_wrapper, kernel_wrapper, write_wrapper);
    }

    return convolved_output;
}


