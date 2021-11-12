from . import fastcorr_cpp
import numpy as np


def short_filter_correlate1D(data_array: np.ndarray,
                             filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs fast single channel single filter 1D correlation, where the filter
        is known to be very short and shorter than the data

    :param data_array: np.ndarray, shape (data_len, ); dtype either np.float64 or np.float32
    :param filter_array: np.ndarray, shape (filter_len, ), where filter_len <= data_len; dtype either
        np.float64 or np.float32
    :return: shape (data_len - filter_len + 1, ) corresponding to a "valid" correlation in 1D
    '''

    # check input for correctness
    if data_array.ndim != 1:
        raise ValueError("data_array must have ndim 1")
    if filter_array.ndim != 1:
        raise ValueError("filter_array must have ndim 1")

    data_len, filter_len = data_array.shape[0], filter_array.shape[0]

    if filter_len > data_len:
        raise ValueError("filter_array must be shorter than data_array")

    filtered_unsqueezed = fastcorr_cpp.batched_filters_batched_data_correlate(data_array[None, :],
                                                                              filter_array[None, :])

    return filtered_unsqueezed.squeeze(0).squeeze(0)


def single_filter_multiple_data_correlate1D(data_array: np.ndarray,
                                            filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs parallel 1D correlations (multiple data, single filter), where the
        filter is known to be very short and shorter than the data

    :param data_array: np.ndarray, shape (batch_data, data_len); dtype either np.float32 or np.float64
    :param filter_array: np.ndarray, shape (filter_len, ), where filter_len <= data_len; dtype either
        np.float64 or np.float32
    :return: shape (batch_data, data_len - filter_len + 1)
    '''
    if data_array.ndim != 2:
        raise ValueError("data_array must have ndim 2")
    if filter_array.ndim != 1:
        raise ValueError("filter_array must have ndim 1")

    filter_len = filter_array.shape[0]
    n_chan_data, data_len = data_array.shape

    if filter_len > data_len:
        raise ValueError("filter_array must be shorter than data_array")

    filtered_unsqueezed = fastcorr_cpp.batched_filters_batched_data_correlate(data_array,
                                                                              filter_array[None, :])

    return filtered_unsqueezed.squeeze(1)


def multiple_filter_single_data_correlate1D(data_array: np.ndarray,
                                            filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs parallel 1D correlations (single data, multiple filters), where the
        filter is known to be very short and shorter than the data

    :param data_array: np.ndarray, shape (data_len, ); dtype either np.float32 or np.float64
    :param filter_array: np.ndarray, shape (n_filters, filter_len), where filter_len <= data_len; dtype either
        np.float64 or np.float32
    :return: shape (n_filters, data_len - filter_len + 1)
    '''

    if data_array.ndim != 1:
        raise ValueError("data_array must have ndim 1")
    if filter_array.ndim != 2:
        raise ValueError("filter_array must have ndim 2")

    n_filters, filter_len = filter_array.shape
    data_len = data_array.shape[0]

    if filter_len > data_len:
        raise ValueError("filter_array must have fewer samples than data_array")

    filtered_unsqueezed = fastcorr_cpp.batched_filters_batched_data_correlate(data_array[None, :],
                                                                              filter_array)

    return filtered_unsqueezed.squeeze(0)


def multiple_filter_multiple_data_correlate1D(data_array: np.ndarray,
                                              filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs parallel 1D correlations (multiple data, multiple filters), where the
        filter is known to be very short and shorter than the data

    :param data_array: np.ndarray, shape (batch_data, data_len); dtype either np.float32 or np.float64
    :param filter_array: np.ndarray, shape (n_filters, filter_len), where filter_len <= data_len; dtype either
        np.float64 or np.float32
    :return: shape (batch_data, n_filters, data_len - filter_len + 1)
    '''
    if data_array.ndim != 2:
        raise ValueError("data_array must have ndim 2")
    if filter_array.ndim != 2:
        raise ValueError("filter_array must have ndim 2")

    data_len, filter_len = data_array.shape[1], filter_array.shape[1]

    if filter_len > data_len:
        raise ValueError("filter_array must have fewer samples than data_array")

    return fastcorr_cpp.batched_filters_batched_data_correlate(data_array,
                                                               filter_array)


def batch_filter_batch_data_channel_correlate1D(data_array: np.ndarray,
                                                filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs a multi-channel correlate, where we correlate each channel
        of data with its own filter. Batched over a set of multichannel filters, and batched over
        a set of multichannel data.

    Conceptually similar to batch_data_batch_filter_multichan_accum_correlate1D, but does not perform
        the accumulation sum over the channels

    :param data_array: shape (batch_data, n_channels, n_samples_data); dtype either np.float32 or np.float64
    :param filter_array: shape (batch_filter, n_channels, n_samples_filters), n_samples_filters < n_samples_data;
        dtype either np.float32 or np.float64;
    :return: shape (batch_data, batch_filter, n_channels, n_samples_data - n_samples_filters + 1)
    '''
    if data_array.ndim != 3:
        raise ValueError("data_array must have ndim 3")
    if filter_array.ndim != 3:
        raise ValueError("filter_array must have ndim 3")

    batch_data, ch_data, data_len = data_array.shape
    batch_filter, ch_data_f, filter_len = filter_array.shape

    if ch_data != ch_data_f:
        raise ValueError("data_array and filter_array must have same number of channels")

    if filter_len > data_len:
        raise ValueError("filter_array must have fewer samples than data_array")

    return fastcorr_cpp.batched_filters_batched_data_channel_correlate(data_array,
                                                                       filter_array)


def batch_filter_single_data_channel_correlate1D(data_array: np.ndarray,
                                                 filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs a multi-channel correlate, where we correlate each channel
        of data with its own filter. Batched over a set of multichannel filters.

    Conceptually similar to batch_filter_multichan_accum_correlate1D, but does not perform
        the accumulation sum over the channels

    :param data_array: shape (n_channels, n_samples_data); dtype either np.float32 or np.float64
    :param filter_array: shape (batch_filter, n_channels, n_samples_filters), n_samples_filters < n_samples_data;
        dtype either np.float32 or np.float64;
    :return: shape (batch_filter, n_channels, n_samples_data - n_samples_filters + 1)
    '''

    if data_array.ndim != 2:
        raise ValueError("data_array must have ndim 2")
    if filter_array.ndim != 3:
        raise ValueError("filter_array must have ndim 3")

    ch_data, data_len = data_array.shape
    batch_filter, ch_data_f, filter_len = filter_array.shape

    if ch_data != ch_data_f:
        raise ValueError("data_array and filter_array must have same number of channels")

    if filter_len > data_len:
        raise ValueError("filter_array must have fewer samples than data_array")

    return fastcorr_cpp.batched_filters_batched_data_channel_correlate(data_array[None, :, :],
                                                                       filter_array).squeeze(0)


def single_filter_batch_data_channel_correlate1D(data_array: np.ndarray,
                                                 filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs a multi-channel correlate, where we correlate each channel
        of data with its own filter. Batched over a set of multichannel data.

    Conceptually similar to batch_data_multichan_accum_correlate1D, but does not perform
        the accumulation sum over the channels

    :param data_array: shape (batch_data, n_channels, n_samples_data); dtype either np.float32 or np.float64
    :param filter_array: shape (n_channels, n_samples_filters), n_samples_filters < n_samples_data;
        dtype either np.float32 or np.float64;
    :return: shape (batch_data, n_channels, n_samples_data - n_samples_filters + 1)
    '''
    if data_array.ndim != 3:
        raise ValueError("data_array must have ndim 3")
    if filter_array.ndim != 2:
        raise ValueError("filter_array must have ndim 2")

    batch_data, ch_data, data_len = data_array.shape
    ch_data_f, filter_len = filter_array.shape

    if ch_data != ch_data_f:
        raise ValueError("data_array and filter_array must have same number of channels")

    if filter_len > data_len:
        raise ValueError("filter_array must have fewer samples than data_array")

    return fastcorr_cpp.batched_filters_batched_data_channel_correlate(data_array,
                                                                       filter_array[None, :, :]).squeeze(1)


def single_filter_single_data_channel_correlate1D(data_array: np.ndarray,
                                                 filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs a multi-channel correlate, where we correlate each channel
        of data with its own filter.

    Conceptually similar to multichan_accum_correlate1D, but does not perform
        the accumulation sum over the channels

    :param data_array: shape (n_channels, n_samples_data); dtype either np.float32 or np.float64
    :param filter_array: shape (n_channels, n_samples_filters), n_samples_filters < n_samples_data;
        dtype either np.float32 or np.float64;
    :return: shape (n_channels, n_samples_data - n_samples_filters + 1)
    '''

    if data_array.ndim != 2:
        raise ValueError("data_array must have ndim 3")
    if filter_array.ndim != 2:
        raise ValueError("filter_array must have ndim 2")

    ch_data, data_len = data_array.shape
    ch_data_f, filter_len = filter_array.shape

    if ch_data != ch_data_f:
        raise ValueError("data_array and filter_array must have same number of channels")

    if filter_len > data_len:
        raise ValueError("filter_array must have fewer samples than data_array")

    return fastcorr_cpp.batched_filters_batched_data_channel_correlate(data_array[None, :, :],
                                                                       filter_array[None, :, :]).squeeze(1).squeeze(0)


def multichan_accum_correlate1D(data_array: np.ndarray,
                                filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs a multi-channel correlate/accumulate, where we correlate each channel
        of data with its own filter, and then sum over of all of the channels

    Useful for template-matching with multiple data channels

    :param data_array: np.ndarray of shape (n_channels, data_len); dtype either np.float64 or np.float32
    :param filter_array: np.ndarray of shape (n_channels, filter_len); dtype either np.float64 or np.float32
        filter_len <= data_len
    :return: np.ndarray, shape (data_len - filter_len + 1, )
    '''
    if data_array.ndim != 2:
        raise ValueError("data_array must have ndim 2")
    if filter_array.ndim != 2:
        raise ValueError("filter_array must have ndim 2")

    n_chan_data, data_len = data_array.shape
    n_chan_filter, filter_len = filter_array.shape

    if n_chan_data != n_chan_filter:
        raise ValueError("data_array must have same number of channesl as filter_array")
    if filter_len > data_len:
        raise ValueError("filter_array must have fewer samples than data_array")

    accum_unsqueezed = fastcorr_cpp.batched_data_batched_filter_multichan_correlate_accum(data_array[None, :, :],
                                                                                          filter_array[None, :, :])

    return accum_unsqueezed.squeeze(0).squeeze(0)


def batch_filter_multichan_accum_correlate1D(data_array: np.ndarray,
                                             filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs a multi-channel correlate/accumulate, where we correlate each channel
        of data with its own filter, and then sum over of all of the channels,
        batched over a set of filters

    Useful for parallel template-matching on multiple data channels, where there are multiple candidate
        templates

    :param data_array: np.ndarray of shape (n_channels, data_len); dtype either np.float64 or np.float32
    :param filter_array: np.ndarray of shape (batch_filter, n_channels, filter_len); dtype either np.float64 or np.float32
        filter_len <= data_len
    :return: np.ndarray, shape (batch_filter, data_len - filter_len + 1)
    '''
    if data_array.ndim != 2:
        raise ValueError("data_array must have ndim 2")
    if filter_array.ndim != 3:
        raise ValueError("filter_array must have ndim 3")

    n_chan_data, data_len = data_array.shape
    batch_n_filter, n_chan_filter, filter_len = filter_array.shape

    if n_chan_data != n_chan_filter:
        raise ValueError("data_array must have same number of channesl as filter_array")
    if filter_len > data_len:
        raise ValueError("filter_array must have fewer samples than data_array")

    accum_unsqueezed = fastcorr_cpp.batched_data_batched_filter_multichan_correlate_accum(data_array[None, :, :],
                                                                                          filter_array)

    return accum_unsqueezed.squeeze(0)


def batch_data_multichan_accum_correlate1D(data_array: np.ndarray,
                                           filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs a multi-channel correlate/accumulate, where we correlate each channel
        of data with its own filter, and then sum over of all of the channels,
        batched over a set of filters

    Useful for parallel template-matching on multiple data channels, where there are multiple candidate
        templates

    :param data_array: np.ndarray of shape (batch_n_data, n_channels, data_len); dtype either np.float64 or np.float32
    :param filter_array: np.ndarray of shape (n_channels, filter_len); dtype either np.float64 or np.float32
        filter_len <= data_len
    :return: np.ndarray, shape (batch_n_data, data_len - filter_len + 1)
    '''
    if data_array.ndim != 3:
        raise ValueError("data_array must have ndim 3")
    if filter_array.ndim != 2:
        raise ValueError("filter_array must have ndim 2")

    batch_n_data, n_chan_data, data_len = data_array.shape
    n_chan_filter, filter_len = filter_array.shape

    if n_chan_data != n_chan_filter:
        raise ValueError("data_array must have same number of channesl as filter_array")
    if filter_len > data_len:
        raise ValueError("filter_array must have fewer samples than data_array")

    accum_unsqueezed = fastcorr_cpp.batched_data_batched_filter_multichan_correlate_accum(data_array,
                                                                                          filter_array[None, :, :])

    return accum_unsqueezed.squeeze(1)


def batch_data_batch_filter_multichan_accum_correlate1D(data_array: np.ndarray,
                                                        filter_array: np.ndarray) -> np.ndarray:
    '''
    Performs a multi-channel correlate/accumulate, where we correlate each channel
        of data with its own filter, and then sum over of all of the channels,
        batched over a set of filters

    Useful for parallel template-matching on multiple data channels, where there are multiple candidate
        templates

    :param data_array: np.ndarray of shape (batch_n_data, n_channels, data_len); dtype either np.float64 or np.float32
    :param filter_array: np.ndarray of shape (n_filters, n_channels, filter_len); dtype either np.float64 or np.float32
        filter_len <= data_len
    :return: np.ndarray, shape (batch_n_data, n_filters, data_len - filter_len + 1)
    '''
    if data_array.ndim != 3:
        raise ValueError("data_array must have ndim 3")
    if filter_array.ndim != 3:
        raise ValueError("filter_array must have ndim 3")

    batch_n_data, n_chan_data, data_len = data_array.shape
    batch_n_filter, n_chan_filter, filter_len = filter_array.shape

    if n_chan_data != n_chan_filter:
        raise ValueError("data_array must have same number of channesl as filter_array")
    if filter_len > data_len:
        raise ValueError("filter_array must have fewer samples than data_array")

    accum_unsqueezed = fastcorr_cpp.batched_data_batched_filter_multichan_correlate_accum(data_array,
                                                                                          filter_array)

    return accum_unsqueezed
