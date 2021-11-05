import numpy as np
from . import imageconv_cpp

def batch_parallel_2Dconv_same(batched_images: np.ndarray,
                               filter_coeffs: np.ndarray,
                               pad_values: float) -> np.ndarray:

    '''
    Performs batched 2D "same" convolution of images in parallel

    :param batched_images: np.ndarray shape (batch, height, width), dtype either np.float32 or np.float64
    :param filter_coeffs: np.ndarray of shape (kern_height, kern_width), dtype either np.float32 or np.float64
    :param pad_values: float, value to pad the border by to produce a "same" convolution
    :return:
    '''

    if batched_images.ndim != 3:
        raise ValueError("batched_images must have ndim 3")
    if filter_coeffs.ndim != 2:
        raise ValueError("filter_coeffs must have ndim 2")

    return imageconv_cpp.batch_smallfilter_2dconv_same(batched_images, filter_coeffs, pad_values)


def batch_parallel_2Dconv_valid(batched_images : np.ndarray,
                                filter_coeffs: np.ndarray) -> np.ndarray:
    '''
    Performs batched 2D "same" convolution of images in parallel

    :param batched_images: np.ndarray shape (batch, height, width), dtype either np.float32 or np.float64
    :param filter_coeffs: np.ndarray of shape (kern_height, kern_width), dtype either np.float32 or np.float64
    :return:
    '''

    if batched_images.ndim != 3:
        raise ValueError("batched_images must have ndim 3")
    if filter_coeffs.ndim != 2:
        raise ValueError("filter_coeffs must have ndim 2")

    return imageconv_cpp.batch_smallfilter_2dconv_shrink(batched_images, filter_coeffs)
