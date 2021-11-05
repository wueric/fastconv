#ifndef IMAGECONV_FAST_2DCONV_H
#define IMAGECONV_FAST_2DCONV_H

#include <string.h>

#ifdef __SSE__
#include <emmintrin.h>
#include <nmmintrin.h>
#include <xmmintrin.h>
#endif

#ifdef __AVX__
#include <immintrin.h>
#endif

#include <stdint.h>
#include "NDContigArrayWrapper.h"


int64_t conv2D_samesize(
        TwoDContigArrayWrapper<float> &input_wrapper,
        TwoDContigArrayWrapper<float> &kernel_wrapper,
        TwoDContigArrayWrapper<float> &output_buffer,
        float pad_val) {

    float *in = input_wrapper.array_ptr;
    float *out = output_buffer.array_ptr;
    float *kernel = kernel_wrapper.array_ptr;

    int64_t data_size_X = input_wrapper.dim1;
    int64_t data_size_Y = input_wrapper.dim0;

    int64_t kernel_size_X = input_wrapper.dim1;
    int64_t kernel_size_Y = input_wrapper.dim0;

    // the x coordinate of the kernel's center
    int64_t kern_cent_X = (kernel_size_X - 1) / 2;
    // the y coordinate of the kernel's center
    int64_t kern_cent_Y = (kernel_size_Y - 1) / 2;

    int64_t in_Y = 2 * kern_cent_Y + data_size_Y;
    int64_t in_X = (2 * kern_cent_X + data_size_X + 3) / 4 * 4;
    float *in_padded = (float *) malloc(in_Y * in_X * sizeof(float));

    int64_t a = 0;
#ifdef __SSE__
    __m128 padding_block = _mm_set1_ps(pad_val);
    for (; a < in_X * in_Y; a += 4) {
        _mm_storeu_ps(in_padded + a, padding_block);
    }
#endif
    for (; a < in_X * in_Y; ++a) {
        *(in_padded + a) = pad_val;
    }

    for (int64_t a = 0; a < data_size_Y; ++a) {
        memcpy(in_padded + (a + kern_cent_Y) * in_X + kern_cent_X, in + a * data_size_X, sizeof(float) * data_size_X);
    }

    float *kern_flipped = (float *) malloc(kernel_size_X * kernel_size_Y * sizeof(float));
    for (int64_t a = 0; a < kernel_size_X * kernel_size_Y; ++a) {
        kern_flipped[a] = kernel[kernel_size_X * kernel_size_Y - a - 1];
    }


    int64_t xMax = kern_cent_X + data_size_X;
    int64_t yMax = kern_cent_Y + data_size_Y;
    for (int64_t y = kern_cent_Y; y < yMax; y++) {
        int64_t x = kern_cent_X;
#ifdef __SSE__
        for (; x <= xMax - 4; x += 4) {
            float *writeAddress = out + (x - kern_cent_X) + (y - kern_cent_Y) * data_size_X;
            __m128 total = _mm_setzero_ps();
            for (int64_t j = 0; j < kernel_size_Y; j++) {
                for (int64_t i = 0; i < kernel_size_X; i++) {
                    __m128 kernel_block = _mm_load1_ps(kern_flipped + i + j * kernel_size_X);
                    float *inputAddress = in_padded + (x + i - kern_cent_X) + (y + j - kern_cent_Y) * in_X;

                    __m128 temp = _mm_loadu_ps(inputAddress);
                    temp = _mm_mul_ps(temp, kernel_block);
                    total = _mm_add_ps(total, temp);
                }
            }
            _mm_storeu_ps(writeAddress, total);
        }
#endif
        for (; x < xMax; ++x) {
            float *writeAddress = out + (x - kern_cent_X) + (y - kern_cent_Y) * data_size_X;
            float total = 0;
            for (int64_t j = 0; j < kernel_size_Y; j++) {
                for (int64_t i = 0; i < kernel_size_X; i++) {
                    float kernel_number = *(kern_flipped + i + j * kernel_size_X);
                    float *inputAddress = in_padded + (x + i - kern_cent_X) + (y + j - kern_cent_Y) * in_X;
                    total += *(inputAddress) * kernel_number;
                }
            }
            *writeAddress = total;
        }
    }

    free(in_padded);
    free(kern_flipped);
    return 0;
}


int64_t conv2D_samesize(
        TwoDContigArrayWrapper<double> &input_wrapper,
        TwoDContigArrayWrapper<double> &kernel_wrapper,
        TwoDContigArrayWrapper<double> &output_buffer,
        double pad_val) {

    double *in = input_wrapper.array_ptr;
    double *out = output_buffer.array_ptr;
    double *kernel = kernel_wrapper.array_ptr;

    int64_t data_size_X = input_wrapper.dim1;
    int64_t data_size_Y = input_wrapper.dim0;

    int64_t kernel_size_X = input_wrapper.dim1;
    int64_t kernel_size_Y = input_wrapper.dim0;

    // the x coordinate of the kernel's center
    int64_t kern_cent_X = (kernel_size_X - 1) / 2;
    // the y coordinate of the kernel's center
    int64_t kern_cent_Y = (kernel_size_Y - 1) / 2;

    int64_t in_Y = 2 * kern_cent_Y + data_size_Y;
    int64_t in_X = (2 * kern_cent_X + data_size_X + 3) / 4 * 4;
    double *in_padded = (double *) malloc(in_Y * in_X * sizeof(double));

    int64_t a = 0;
#ifdef __AVX__
    __m256d padding_block = _mm256_set1_pd(pad_val);
    for (; a < in_X * in_Y; a += 4) {
        _mm256_storeu_pd(in_padded + a, padding_block);
    }
#endif
    for (; a < in_X * in_Y; ++a) {
        *(in_padded + a) = pad_val;
    }

    for (int64_t a = 0; a < data_size_Y; ++a) {
        memcpy(in_padded + (a + kern_cent_Y) * in_X + kern_cent_X, in + a * data_size_X, sizeof(double) * data_size_X);
    }

    double *kern_flipped = (double *) malloc(kernel_size_X * kernel_size_Y * sizeof(double));
    for (int64_t a = 0; a < kernel_size_X * kernel_size_Y; ++a) {
        kern_flipped[a] = kernel[kernel_size_X * kernel_size_Y - a - 1];
    }

    int64_t xMax = kern_cent_X + data_size_X;
    int64_t yMax = kern_cent_Y + data_size_Y;
    for (int64_t y = kern_cent_Y; y < yMax; y++) {
        int64_t x = kern_cent_X;
#ifdef __AVX__
        for (; x <= xMax - 4; x += 4) {
            double *writeAddress = out + (x - kern_cent_X) + (y - kern_cent_Y) * data_size_X;
            __m256d total = _mm256_setzero_pd();
            for (int64_t j = 0; j < kernel_size_Y; j++) {
                for (int64_t i = 0; i < kernel_size_X; i++) {
                    __m256d kernel_block = _mm256_broadcast_sd(kern_flipped + i + j * kernel_size_X);
                    double *inputAddress = in_padded + (x + i - kern_cent_X) + (y + j - kern_cent_Y) * in_X;

                    __m256d temp = _mm256_loadu_pd(inputAddress);
                    temp = _mm256_mul_pd(temp, kernel_block);
                    total = _mm256_add_pd(total, temp);
                }
            }
            _mm256_storeu_pd(writeAddress, total);
        }
#endif
        for (; x < xMax; ++x) {
            double *writeAddress = out + (x - kern_cent_X) + (y - kern_cent_Y) * data_size_X;
            double total = 0;
            for (int64_t j = 0; j < kernel_size_Y; j++) {
                for (int64_t i = 0; i < kernel_size_X; i++) {
                    double kernel_number = *(kern_flipped + i + j * kernel_size_X);
                    double *inputAddress = in_padded + (x + i - kern_cent_X) + (y + j - kern_cent_Y) * in_X;
                    total += *(inputAddress) * kernel_number;
                }
            }
            *writeAddress = total;
        }
    }

    free(in_padded);
    free(kern_flipped);
    return 0;
}

int64_t conv2D_shrinksize(
        TwoDContigArrayWrapper<float> &input_wrapper,
        TwoDContigArrayWrapper<float> &kernel_wrapper,
        TwoDContigArrayWrapper<float> &output_buffer) {

    float *in = input_wrapper.array_ptr;
    float *out = output_buffer.array_ptr;
    float *kernel = kernel_wrapper.array_ptr;

    int64_t data_size_X = input_wrapper.dim1;
    int64_t data_size_Y = input_wrapper.dim0;

    int64_t kernel_size_X = input_wrapper.dim1;
    int64_t kernel_size_Y = input_wrapper.dim0;

    // the x coordinate of the kernel's center
    int64_t kern_cent_X = (kernel_size_X - 1) / 2;
    // the y coordinate of the kernel's center
    int64_t kern_cent_Y = (kernel_size_Y - 1) / 2;

    // flip and copy the kernel
    float *kern_flipped = (float *) malloc(kernel_size_X * kernel_size_Y * sizeof(float));
    for (int64_t a = 0; a < kernel_size_X * kernel_size_Y; ++a) {
        kern_flipped[a] = kernel[kernel_size_X * kernel_size_Y - a - 1];
    }

    int64_t xMax = data_size_X - kern_cent_X;
    int64_t yMax = data_size_Y - kern_cent_Y;

    int64_t output_size_X = data_size_X - kern_cent_X * 2;
    int64_t output_size_Y = data_size_Y - kern_cent_Y * 2;

    for (int64_t y = kern_cent_Y; y < yMax; y++) {
        int64_t x = kern_cent_X;
#ifdef __SSE__
        for (; x <= xMax - 4; x += 4) {
            float *writeAddress = out + (x - kern_cent_X) + (y - kern_cent_Y) * output_size_X;
            __m128 total = _mm_setzero_ps();
            for (int64_t j = 0; j < kernel_size_Y; ++j) {
                for (int64_t i = 0; i < kernel_size_X; ++i) {
                    __m128 kernel_block = _mm_load1_ps(kern_flipped + i + j * kernel_size_X);
                    float *inputAddress = in + (x + i - kern_cent_X) + (y + j - kern_cent_Y) * data_size_X;

                    __m128 temp = _mm_loadu_ps(inputAddress);
                    temp = _mm_mul_ps(temp, kernel_block);
                    total = _mm_add_ps(total, temp);
                }
            }
            _mm_storeu_ps(writeAddress, total);
        }
#endif
        for (; x < xMax; ++x) {
            float *writeAddress = out + (x - kern_cent_X) + (y - kern_cent_Y) * output_size_X;
            float total = 0;
            for (int64_t j = 0; j < kernel_size_Y; j++) {
                for (int64_t i = 0; i < kernel_size_X; i++) {
                    float kernel_number = *(kern_flipped + i + j * kernel_size_X);
                    float *inputAddress = in + (x + i - kern_cent_X) + (y + j - kern_cent_Y) * data_size_X;
                    total += *(inputAddress) * kernel_number;
                }
            }
            *writeAddress = total;
        }
    }

    free(kern_flipped);
    return 0;
}


int64_t conv2D_shrinksize(
        TwoDContigArrayWrapper<double> &input_wrapper,
        TwoDContigArrayWrapper<double> &kernel_wrapper,
        TwoDContigArrayWrapper<double> &output_buffer) {

    double *in = input_wrapper.array_ptr;
    double *out = output_buffer.array_ptr;
    double *kernel = kernel_wrapper.array_ptr;

    int64_t data_size_X = input_wrapper.dim1;
    int64_t data_size_Y = input_wrapper.dim0;

    int64_t kernel_size_X = input_wrapper.dim1;
    int64_t kernel_size_Y = input_wrapper.dim0;

    // the x coordinate of the kernel's center
    int64_t kern_cent_X = (kernel_size_X - 1) / 2;
    // the y coordinate of the kernel's center
    int64_t kern_cent_Y = (kernel_size_Y - 1) / 2;

    // flip and copy the kernel
    double *kern_flipped = (double *) malloc(kernel_size_X * kernel_size_Y * sizeof(double));
    for (int64_t a = 0; a < kernel_size_X * kernel_size_Y; ++a) {
        kern_flipped[a] = kernel[kernel_size_X * kernel_size_Y - a - 1];
    }

    int64_t xMax = data_size_X - kern_cent_X;
    int64_t yMax = data_size_Y - kern_cent_Y;

    int64_t output_size_X = data_size_X - kern_cent_X * 2;
    int64_t output_size_Y = data_size_Y - kern_cent_Y * 2;

    for (int64_t y = kern_cent_Y; y < yMax; y++) {
        int64_t x = kern_cent_X;
#ifdef __AVX__
        for (; x <= xMax - 4; x += 4) {
            double *writeAddress = out + (x - kern_cent_X) + (y - kern_cent_Y) * output_size_X;
            __m256d total = _mm256_setzero_pd();
            for (int64_t j = 0; j < kernel_size_Y; ++j) {
                for (int64_t i = 0; i < kernel_size_X; ++i) {
                    __m256d kernel_block = _mm256_broadcast_sd(kern_flipped + i + j * kernel_size_X);
                    double *inputAddress = in + (x + i - kern_cent_X) + (y + j - kern_cent_Y) * data_size_X;

                    __m256d temp = _mm256_loadu_pd(inputAddress);
                    temp = _mm256_mul_pd(temp, kernel_block);
                    total = _mm256_add_pd(total, temp);
                }
            }
            _mm256_storeu_pd(writeAddress, total);
        }
#endif
        for (; x < xMax; ++x) {
            double *writeAddress = out + (x - kern_cent_X) + (y - kern_cent_Y) * output_size_X;
            double total = 0;
            for (int64_t j = 0; j < kernel_size_Y; j++) {
                for (int64_t i = 0; i < kernel_size_X; i++) {
                    double kernel_number = *(kern_flipped + i + j * kernel_size_X);
                    double *inputAddress = in + (x + i - kern_cent_X) + (y + j - kern_cent_Y) * data_size_X;
                    total += *(inputAddress) * kernel_number;
                }
            }
            *writeAddress = total;
        }
    }

    free(kern_flipped);
    return 0;
}

#endif
