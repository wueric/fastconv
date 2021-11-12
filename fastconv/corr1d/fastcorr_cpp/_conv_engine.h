#ifndef CONV_ENGINE_H
#define CONV_ENGINE_H

#define SWITCH_TO_SSE_SIZE 256

#include <stdlib.h>
#include <stdint.h>
#ifdef __AVX__
#include <immintrin.h>
#endif
#include <string.h>
#include <math.h>
#include "NDContigArrayWrapper.h"

__m256 _oct_unrolled_kernel_correlate(
        float *long_array,
        const OneDContigArrayWrapper<float> &kernel_wrapper) {

    /*
     * Computes eight entries of the cross-correlation simultaneously
     * Implicitly assumes that we will not go out-of-bounds for long_array
     */

    size_t i = 0;

    float *small_kernel = kernel_wrapper.array_ptr;
    const size_t kernel_length = kernel_wrapper.dim0;

    __m256 from_long_array, from_kernel;

#ifdef __AVX2__

    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    for (; i < kernel_length - 4; i += 4) {
        from_kernel = _mm256_broadcast_ss(small_kernel+i);
        from_long_array = _mm256_loadu_ps(long_array+i);
        acc0 = _mm256_fmadd_ps(from_kernel, from_long_array, acc0);

        from_kernel = _mm256_broadcast_ss(small_kernel+i+1);
        from_long_array = _mm256_loadu_ps(long_array+i+1);
        acc1 = _mm256_fmadd_ps(from_kernel, from_long_array, acc1);

        from_kernel = _mm256_broadcast_ss(small_kernel+i+2);
        from_long_array = _mm256_loadu_ps(long_array+i+2);
        acc2 = _mm256_fmadd_ps(from_kernel, from_long_array, acc2);

        from_kernel = _mm256_broadcast_ss(small_kernel+i+3);
        from_long_array = _mm256_loadu_ps(long_array+i+3);
        acc3 = _mm256_fmadd_ps(from_kernel, from_long_array, acc3);
    }

    acc0 = acc0 + acc1 + acc2 + acc3;

    __m256 prod;
    for (; i < kernel_length; ++i) {
        from_kernel = _mm256_broadcast_ss(small_kernel+i);
        from_long_array = _mm256_loadu_ps(long_array+i);

        prod = _mm256_mul_ps(from_kernel, from_long_array);
        acc0 = _mm256_add_ps(prod, acc0);
    }

    return acc0;
#else

    __m256 quad_accumulator = _mm256_setzero_ps();
    __m256 prod;

    for (; i < kernel_length - 4; i += 4) {
        for (int64_t j = 0; j < 4; ++j) {
            from_kernel = _mm256_broadcast_ss(small_kernel + i + j);
            from_long_array = _mm256_loadu_ps(long_array + i + j);
            prod = _mm256_mul_ps(from_kernel, from_long_array);
            quad_accumulator = _mm256_add_ps(prod, quad_accumulator);
        }
    }

    for (; i < kernel_length; ++i) {
        from_kernel = _mm256_broadcast_ss(small_kernel + i);
        from_long_array = _mm256_loadu_ps(long_array + i);

        prod = _mm256_mul_ps(from_kernel, from_long_array);
        quad_accumulator = _mm256_add_ps(prod, quad_accumulator);
    }

    return quad_accumulator;

#endif
}

float _single_prec_dot_product(float *a, float *b, const size_t length) {
    float final_answer = 0.0;
    for (size_t i = 0; i < length; ++i) final_answer += ((*(a + i)) * (*(b + i)));
    return final_answer;
}


__m256d _quad_unrolled_kernel_correlate(
        double *long_array,
        const OneDContigArrayWrapper<double> &kernel_wrapper) {

    size_t i = 0;

    double *small_kernel = kernel_wrapper.array_ptr;
    const size_t kernel_length = kernel_wrapper.dim0;

    __m256d from_long_array, from_kernel;


#ifdef __AVX2__

    __m256d acc0 = _mm256_setzero_pd();
    __m256d acc1 = _mm256_setzero_pd();
    __m256d acc2 = _mm256_setzero_pd();
    __m256d acc3 = _mm256_setzero_pd();

    for (; i < kernel_length - 4; i += 4) {

        from_kernel = _mm256_broadcast_sd(small_kernel+i);
        from_long_array = _mm256_loadu_pd(long_array+i);
        acc0 = _mm256_fmadd_pd(from_kernel, from_long_array, acc0);

        from_kernel = _mm256_broadcast_sd(small_kernel+i+1);
        from_long_array = _mm256_loadu_pd(long_array+i+1);
        acc1 = _mm256_fmadd_pd(from_kernel, from_long_array, acc1);

        from_kernel = _mm256_broadcast_sd(small_kernel+i+2);
        from_long_array = _mm256_loadu_pd(long_array+i+2);
        acc2 = _mm256_fmadd_pd(from_kernel, from_long_array, acc2);

        from_kernel = _mm256_broadcast_sd(small_kernel+i+3);
        from_long_array = _mm256_loadu_pd(long_array+i+3);
        acc3 = _mm256_fmadd_pd(from_kernel, from_long_array, acc3);
    }

    __m256d prod;
    for (; i < kernel_length; ++i) {
        from_kernel = _mm256_broadcast_sd(small_kernel+i);
        from_long_array = _mm256_loadu_pd(long_array+i);

        prod = _mm256_mul_pd(from_kernel, from_long_array);
        acc0 = _mm256_add_pd(prod, acc0);
    }
    return acc0 + acc1 + acc2 + acc3;
#else

    __m256d quad_accumulator = _mm256_setzero_pd();
    __m256d prod;

    for (; i < kernel_length - 4; i += 4) {
        for (int64_t j = 0; j < 4; ++j) {
            from_kernel = _mm256_broadcast_sd(small_kernel + i + j);
            from_long_array = _mm256_loadu_pd(long_array + i + j);
            prod = _mm256_mul_pd(from_kernel, from_long_array);
            quad_accumulator = _mm256_add_pd(prod, quad_accumulator);
        }
        
    }

    for (; i < kernel_length; ++i) {
        from_kernel = _mm256_broadcast_sd(small_kernel + i);
        from_long_array = _mm256_loadu_pd(long_array + i);

        prod = _mm256_mul_pd(from_kernel, from_long_array);
        quad_accumulator = _mm256_add_pd(prod, quad_accumulator);
    }

    return quad_accumulator;
#endif
}


double _dot_product(double *a, double *b, const int64_t length) {
    double final_answer = 0.0;
    for (int64_t i = 0; i < length; ++i) final_answer += ((*(a + i)) * (*(b + i)));
    return final_answer;
}


void correlate1D(
        OneDContigArrayWrapper<float> &raw_data_wrapper,
        OneDContigArrayWrapper<float> &kernel_wrapper,
        OneDContigArrayWrapper<float> &output_wrapper) {

    const int64_t n_samples_taps = kernel_wrapper.dim0;
    const int64_t n_samples_raw_data = raw_data_wrapper.dim0;
    const int64_t output_buffer_size = output_wrapper.dim0;

    if (output_buffer_size < n_samples_raw_data - n_samples_taps + 1) {
        throw std::invalid_argument("1D convolution output buffer too small");
    }

    float *write_offset = output_wrapper.array_ptr;
    float *filter_taps_array = kernel_wrapper.array_ptr;
    float *raw_data_ptr = raw_data_wrapper.array_ptr;

    int64_t j = 0;
#ifdef __AVX__
    __m256 acc;
    for (; j < n_samples_raw_data - n_samples_taps + 1 - 8; j += 8) {

        acc = _oct_unrolled_kernel_correlate(raw_data_ptr + j,
                                             kernel_wrapper);
        _mm256_storeu_ps(write_offset + j, acc);
    }
#endif

    // now clean up the remainder of the data
    for (; j < n_samples_raw_data - n_samples_taps + 1; ++j) {
        *(write_offset + j) = _single_prec_dot_product(raw_data_ptr + j,
                                                       filter_taps_array,
                                                       n_samples_taps);
    }
}

void correlate_accum_over_channels(
        TwoDContigArrayWrapper<float> &raw_data_wrapper,
        TwoDContigArrayWrapper<float> &kernel_wrapper,
        OneDContigArrayWrapper<float> &output_wrapper) {

    const int64_t n_taps = kernel_wrapper.dim1;
    const int64_t n_chan_filter = kernel_wrapper.dim0;

    const int64_t n_chan_data = raw_data_wrapper.dim0;
    const int64_t n_samples_data = raw_data_wrapper.dim1;

    const int64_t output_buffer_size = output_wrapper.dim0;

    if (output_buffer_size < n_samples_data - n_taps + 1) {
        throw std::invalid_argument("Convolution output buffer too small");
    }

    float *filter_taps_array = kernel_wrapper.array_ptr;
    float *raw_data_ptr = raw_data_wrapper.array_ptr;
    float *output_ptr = output_wrapper.array_ptr;

    int64_t j = 0;
#ifdef __AVX__
    __m256 acc, temp;
    for (; j < n_samples_data - n_taps + 1 - 8; j += 8) {

        acc = _mm256_setzero_ps();

        for (int64_t ch = 0; ch < n_chan_filter; ++ch) {

            temp = _oct_unrolled_kernel_correlate(
                    raw_data_ptr + n_samples_data * ch + j,
                    OneDContigArrayWrapper<float>(filter_taps_array + ch * n_taps, n_taps));

            acc += temp;
        }

        _mm256_storeu_ps(output_ptr + j, acc);
    }
#endif

    float rem_acc, temp2;
    for (; j < n_samples_data - n_taps + 1; ++j) {

        rem_acc = 0.0;

        for (int64_t ch = 0; ch < n_chan_filter; ++ch) {

            temp2 = _single_prec_dot_product(
                    raw_data_ptr + ch * n_samples_data + j,
                    filter_taps_array + ch * n_taps,
                    n_taps);

            rem_acc += temp2;

        }
        *(output_ptr + j) = rem_acc;
    }
}


void correlate1D(
        OneDContigArrayWrapper<double> &raw_data_wrapper,
        OneDContigArrayWrapper<double> &kernel_wrapper,
        OneDContigArrayWrapper<double> &output_wrapper) {

    const int64_t n_samples_taps = kernel_wrapper.dim0;
    const int64_t n_samples_raw_data = raw_data_wrapper.dim0;
    const int64_t output_buffer_size = output_wrapper.dim0;

    if (output_buffer_size < n_samples_raw_data - n_samples_taps + 1) {
        throw std::invalid_argument("1D convolution output buffer too small");
    }

    double *write_offset = output_wrapper.array_ptr;
    double *filter_taps_array = kernel_wrapper.array_ptr;
    double *raw_data_ptr = raw_data_wrapper.array_ptr;

    int64_t j = 0;
#ifdef __AVX__

    __m256d acc;

    // fast correlate over the full length of the kernel, up to a certain point
    // in the data
    for (; j < n_samples_raw_data - n_samples_taps + 1 - 4; j += 4) {

        acc = _quad_unrolled_kernel_correlate(raw_data_ptr + j,
                                              kernel_wrapper);
        _mm256_storeu_pd(write_offset + j, acc);

    }

#endif

    // now clean up the remainder of the data
    for (; j < n_samples_raw_data - n_samples_taps + 1; ++j) {

        *(write_offset + j) = _dot_product(raw_data_ptr + j,
                                                    filter_taps_array,
                                                    n_samples_taps);
    }
}


void correlate_accum_over_channels(
        TwoDContigArrayWrapper<double> &raw_data_wrapper,
        TwoDContigArrayWrapper<double> &kernel_wrapper,
        OneDContigArrayWrapper<double> &output_wrapper) {

    const int64_t n_taps = kernel_wrapper.dim1;
    const int64_t n_chan_filter = kernel_wrapper.dim0;

    const int64_t n_chan_data = raw_data_wrapper.dim0;
    const int64_t n_samples_data = raw_data_wrapper.dim1;

    const int64_t output_buffer_size = output_wrapper.dim0;

    if (output_buffer_size < n_samples_data - n_taps + 1) {
        throw std::invalid_argument("Convolution output buffer too small");
    }

    double *filter_taps_array = kernel_wrapper.array_ptr;
    double *raw_data_ptr = raw_data_wrapper.array_ptr;
    double *output_ptr = output_wrapper.array_ptr;

    int64_t j = 0;
#ifdef __AVX__
    __m256d acc, temp;
    for (; j < n_samples_data - n_taps + 1 - 4; j += 4) {

        acc = _mm256_setzero_pd();

        for (int64_t ch = 0; ch < n_chan_filter; ++ch) {

            temp = _quad_unrolled_kernel_correlate(
                    raw_data_ptr + n_samples_data * ch + j,
                    OneDContigArrayWrapper<double>(filter_taps_array + ch * n_taps, n_taps));

            acc += temp;
        }

        _mm256_storeu_pd(output_ptr + j, acc);
    }
#endif

    double rem_acc, temp2;
    for (; j < n_samples_data - n_taps + 1; ++j) {

        rem_acc = 0.0;

        for (int64_t ch = 0; ch < n_chan_filter; ++ch) {

            temp2 = _dot_product(raw_data_ptr + ch * n_samples_data + j,
                                    filter_taps_array + ch * n_taps,
                                    n_taps);

	    rem_acc += temp2;

        }
        *(output_ptr + j) = rem_acc;
    }
}

#endif
