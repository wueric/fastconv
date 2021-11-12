//
// Created by Eric Wu on 11/4/21.
//

#ifndef FASTCONV__FAST1DCORR_H
#define FASTCONV__FAST1DCORR_H

#include "NDContigArrayWrapper.h"
#include "_conv_engine.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <stdlib.h>
#include <stdint.h>
#include <immintrin.h>

#if defined(ENABLE_OPENMP)
#include <omp.h>
#else
typedef int omp_int_t;

inline omp_int_t omp_get_thread_num() { return 0; }

inline omp_int_t omp_get_max_threads() { return 1; }

inline void omp_set_num_threads(int num_threads) { return; }

#endif

#define DATA_SAMPLES_PARALLEL_THRESH 512

namespace py=pybind11;

template<class T>
py::array_t<T, py::array::c_style | py::array::forcecast> batched_filters_batched_data_correlate(
        py::array_t<T, py::array::c_style | py::array::forcecast> &batched_data,
        py::array_t<T, py::array::c_style | py::array::forcecast> &batched_filters) {

    /*
     * Correlate many channels of data with a single shared filter
     */

    py::buffer_info raw_data_info = batched_data.request();
    T *raw_data_ptr = static_cast<T *> (raw_data_info.ptr);
    if (raw_data_info.ndim != 2) {
        throw std::invalid_argument("Data must have dim 2");
    }

    const int64_t n_samples_raw_data = raw_data_info.shape[1];
    const int64_t n_channels_raw_data = raw_data_info.shape[0];

    py::buffer_info filter_taps_info = batched_filters.request();
    T *filter_taps_array = static_cast<T *> (filter_taps_info.ptr);
    if (filter_taps_info.ndim != 2) {
        throw std::invalid_argument("Batched filter must have dim 2");
    }
    const int64_t n_samples_taps = filter_taps_info.shape[1];
    const int64_t batch_n_filters = filter_taps_info.shape[0];

    if (n_samples_taps > n_samples_raw_data) {
        throw std::invalid_argument("Filter must have fewer samples than data");
    }

    const int64_t n_samples_output = n_samples_raw_data - n_samples_taps + 1;

    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(T),     /* Size of one item */
            py::format_descriptor<T>::value, /* Buffer format */
            3,          /* How many dimensions? */
            {n_channels_raw_data, batch_n_filters, n_samples_output},  /* Number of elements for each dimension */
            {sizeof(T) * n_samples_output * batch_n_filters, sizeof(T) * n_samples_output, sizeof(T)}
            /* Strides for each dimension */
    );

    py::array_t <T> convolved_output = py::array_t<T>(output_buffer_info);
    py::buffer_info output_info = convolved_output.request();
    T *output_data_ptr = static_cast<T *> (output_info.ptr);

    omp_set_num_threads(16);
#pragma omp parallel for collapse(2) if(n_channels_raw_data * batch_n_filters > 64 && n_samples_raw_data > DATA_SAMPLES_PARALLEL_THRESH)
    for (int64_t data_ch = 0; data_ch < n_channels_raw_data; ++data_ch) {
        for (int64_t batch_ch = 0; batch_ch < batch_n_filters; ++batch_ch) {

            OneDContigArrayWrapper<T> chan_data_wrapper = OneDContigArrayWrapper<T>(
                    raw_data_ptr + data_ch * n_samples_raw_data,
                    n_samples_raw_data);

            OneDContigArrayWrapper<T> kernel_wrapper = OneDContigArrayWrapper<T>(
                    filter_taps_array + batch_ch * n_samples_taps,
                    n_samples_taps);

            size_t output_offset = (data_ch * batch_n_filters + batch_ch) * n_samples_output;
            OneDContigArrayWrapper<T> output_wrapper = OneDContigArrayWrapper<T>(
                    output_data_ptr + output_offset,
                    n_samples_output);

            correlate1D(chan_data_wrapper, kernel_wrapper, output_wrapper);
        }
    }

    return convolved_output;
}


template<class T>
py::array_t<T, py::array::c_style | py::array::forcecast> batched_filters_batched_data_channel_correlate(
        py::array_t<T, py::array::c_style | py::array::forcecast> &batched_multichan_data,
        py::array_t<T, py::array::c_style | py::array::forcecast> &batched_multichan_filters) {
    /*
     * Function that does batched multi-channel correlations, where the channels are the same
     *      for both the data AND the filters
     *
     * Conceptually very similar to batched_data_batched_filter_multichan_correlate_accum,
     *      but this doesn't do accumulation over the channels
     *
     * @param batched_multichan_data: Contiguous numpy array buffer,
     *      shape (batch_data, channels, n_samples_data)
     *
     * @param batched_multichan_filters: Contiguous numpy array buffer,
     *      shape (batch_filters, channels, n_samples_filters)
     *      Assumed to be shorter in number of samples than batched_multichan_data
     *
     * @returns: Contiguous numpy array buffer,
     *      shape (batch_data, batch_filters, channels, n_samples_data - n_samples_filters + 1)
     */

    py::buffer_info multichan_data_info = batched_multichan_data.request();
    T *multichan_data_ptr = static_cast<T *> (multichan_data_info.ptr);
    const int64_t n_samples_raw_data = multichan_data_info.shape[2];
    const int64_t n_channels_raw_data = multichan_data_info.shape[1];
    const int64_t n_batch_data = multichan_data_info.shape[0];

    py::buffer_info filter_taps_info = batched_multichan_filters.request();
    T *filter_taps_array = static_cast<T *> (filter_taps_info.ptr);
    const int64_t n_batch_filters = filter_taps_info.shape[0];
    const int64_t n_channels_filters = filter_taps_info.shape[1];
    const int64_t n_taps_filters = filter_taps_info.shape[2];

    if (n_channels_filters != n_channels_raw_data) {
        throw std::invalid_argument("Number of data and kernel channels must match");
    }

    const int64_t n_samples_output = n_samples_raw_data - n_taps_filters + 1;

    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(T),     /* Size of one item */
            py::format_descriptor<T>::value, /* Buffer format */
            4,          /* How many dimensions? */
            {n_batch_data, n_batch_filters, n_channels_filters,
             n_samples_output},  /* Number of elements for each dimension */
            {sizeof(T) * n_samples_output * n_channels_filters * n_batch_filters,
             sizeof(T) * n_samples_output * n_channels_filters,
             sizeof(T) * n_samples_output,
             sizeof(T)} /* Strides for each dimension */
    );

    py::array_t <T> convolved_output = py::array_t<T>(output_buffer_info);
    py::buffer_info output_info = convolved_output.request();
    T *output_data_ptr = static_cast<T *> (output_info.ptr);

    omp_set_num_threads(16);
#pragma omp parallel for collapse(3) if(n_batch_filters * n_batch_data * n_channels_filters > 64 && n_samples_raw_data > DATA_SAMPLES_PARALLEL_THRESH)
    for (int64_t b_data = 0; b_data < n_batch_data; ++b_data) {
        for (int64_t b_filter = 0; b_filter < n_batch_filters; ++b_filter) {
            for (int64_t b_ch = 0; b_ch < n_channels_filters; ++b_ch) {

                int64_t data_offset =
                        b_data * (n_channels_filters * n_samples_raw_data) + b_ch * n_samples_raw_data;
                OneDContigArrayWrapper<T> data_wrapper = OneDContigArrayWrapper<T>(
                        multichan_data_ptr + data_offset,
                        n_samples_raw_data);

                int64_t filter_offset = b_filter * (n_channels_filters * n_taps_filters) + b_ch * n_taps_filters;
                OneDContigArrayWrapper<T> kernel_wrapper = OneDContigArrayWrapper<T>(
                        filter_taps_array + filter_offset,
                        n_taps_filters);

                int64_t output_offset = b_data * (n_samples_output * n_channels_filters * n_batch_filters) +
                        b_filter * (n_samples_output * n_channels_filters) + b_ch * n_samples_output;
                OneDContigArrayWrapper<T> output_wrapper = OneDContigArrayWrapper<T>(
                        output_data_ptr + output_offset,
                        n_samples_output);

                correlate1D(data_wrapper, kernel_wrapper, output_wrapper);
            }
        }
    }

    return convolved_output;
}


template<class T>
py::array_t<T, py::array::c_style | py::array::forcecast> batched_data_batched_filter_multichan_correlate_accum(
        py::array_t<T, py::array::c_style | py::array::forcecast> &batched_multichan_data,
        py::array_t<T, py::array::c_style | py::array::forcecast> &batched_multichan_filters) {

    py::buffer_info multichan_data_info = batched_multichan_data.request();
    T *multichan_data_ptr = static_cast<T *> (multichan_data_info.ptr);
    const int64_t n_samples_raw_data = multichan_data_info.shape[2];
    const int64_t n_channels_raw_data = multichan_data_info.shape[1];
    const int64_t n_batch_data = multichan_data_info.shape[0];

    py::buffer_info filter_taps_info = batched_multichan_filters.request();
    T *filter_taps_array = static_cast<T *> (filter_taps_info.ptr);
    const int64_t n_batch_filters = filter_taps_info.shape[0];
    const int64_t n_channels_filters = filter_taps_info.shape[1];
    const int64_t n_taps_filters = filter_taps_info.shape[2];

    if (n_channels_filters != n_channels_raw_data) {
        throw std::invalid_argument("Number of data and kernel channels must match");
    }

    const int64_t n_samples_output = n_samples_raw_data - n_taps_filters + 1;

    auto output_buffer_info = py::buffer_info(
            nullptr,            /* Pointer to data (nullptr -> ask NumPy to allocate!) */
            sizeof(T),     /* Size of one item */
            py::format_descriptor<T>::value, /* Buffer format */
            3,          /* How many dimensions? */
            {n_batch_data, n_batch_filters, n_samples_output},  /* Number of elements for each dimension */
            {n_samples_output * sizeof(T) * n_batch_filters, n_samples_output * sizeof(T), sizeof(T)}
            /* Strides for each dimension */
    );

    py::array_t <T> convolved_output = py::array_t<T>(output_buffer_info);
    py::buffer_info output_info = convolved_output.request();
    T *output_data_ptr = static_cast<T *> (output_info.ptr);

    omp_set_num_threads(16);
#pragma omp parallel for collapse(2) if(n_batch_filters * n_batch_data > 64 && n_samples_raw_data > DATA_SAMPLES_PARALLEL_THRESH)
    for (int64_t b_data = 0; b_data < n_batch_data; ++b_data) {
        for (int64_t b_filter = 0; b_filter < n_batch_filters; ++b_filter) {

            T *data_offset_ptr = multichan_data_ptr + b_data * (n_channels_raw_data * n_samples_raw_data);
            TwoDContigArrayWrapper<T> data_wrapper = TwoDContigArrayWrapper<T>(
                    data_offset_ptr,
                    n_channels_raw_data,
                    n_samples_raw_data);

            T *kernel_offset_ptr = filter_taps_array + b_filter * (n_channels_filters * n_taps_filters);
            TwoDContigArrayWrapper<T> filter_wrapper = TwoDContigArrayWrapper<T>(
                    kernel_offset_ptr,
                    n_channels_filters,
                    n_taps_filters);

            T *write_offset_ptr = output_data_ptr + (b_data * n_batch_filters + b_filter) * n_samples_output;
            OneDContigArrayWrapper<T> output_wrapper = OneDContigArrayWrapper<T>(
                    write_offset_ptr,
                    n_samples_output);

            correlate_accum_over_channels(data_wrapper, filter_wrapper, output_wrapper);
        }
    }

    return convolved_output;
}


#endif //FASTCONV__FAST1DCORR_H
