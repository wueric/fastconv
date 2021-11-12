from fastconv import corr1d
import numpy as np

import os

TESTCASE_PATH = 'testcases'
MULTIDATA_MULTICHAN_FNAME = 'multidata_multichan.npy'
MULTICHAN_FILTERS_FNAME = 'multifilters.npy'

FILTER_PATH = os.path.join(TESTCASE_PATH, MULTICHAN_FILTERS_FNAME)
DATA_PATH = os.path.join(TESTCASE_PATH, MULTIDATA_MULTICHAN_FNAME)

FILTERS = np.load(FILTER_PATH)
DATA = np.load(DATA_PATH)


def test_corr1D_AA_double():
    test_data = DATA[0, 35, :].astype(np.float64)
    test_filter = FILTERS[0, 35, :].astype(np.float64)

    comparison = np.correlate(test_data, test_filter, mode='valid')  # type: np.ndarray
    my_output = corr1d.short_filter_correlate1D(test_data, test_filter)  # type: np.ndarray

    assert my_output.dtype == np.float64, 'data type should be np.float64'

    assert comparison.shape == my_output.shape, 'shape is {0}, should be {1}'.format(my_output.shape,
                                                                                     comparison.shape)

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_corr1D_AA_float():
    test_data = DATA[0, 35, :].astype(np.float32)
    test_filter = FILTERS[0, 35, :].astype(np.float32)

    comparison = np.correlate(test_data, test_filter, mode='valid')
    my_output = corr1d.short_filter_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float32, 'data type should be np.float32'

    assert comparison.shape == my_output.shape, 'shape is {0}, should be {1}'.format(my_output.shape,
                                                                                     comparison.shape)

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_corr1D_AB_double():
    test_data = DATA[0, 35, 1000:2001].astype(np.float64)
    test_filter = FILTERS[0, 35, :].astype(np.float64)

    comparison = np.correlate(test_data, test_filter, mode='valid')  # type: np.ndarray
    my_output = corr1d.short_filter_correlate1D(test_data, test_filter)  # type: np.ndarray

    assert my_output.dtype == np.float64, 'data type should be np.float64'

    assert comparison.shape == my_output.shape, 'shape is {0}, should be {1}'.format(my_output.shape,
                                                                                     comparison.shape)

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_corr1D_AB_float():
    test_data = DATA[0, 35, 1000:2001].astype(np.float32)
    test_filter = FILTERS[0, 35, :].astype(np.float32)

    comparison = np.correlate(test_data, test_filter, mode='valid')
    my_output = corr1d.short_filter_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float32, 'data type should be np.float32'

    assert comparison.shape == my_output.shape, 'shape is {0}, should be {1}'.format(my_output.shape,
                                                                                     comparison.shape)

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_corr1D_AC_double():
    test_data = FILTERS[0, 35, :].astype(np.float64)
    test_filter = FILTERS[0, 35, :].astype(np.float64)

    comparison = np.correlate(test_data, test_filter, mode='valid')  # type: np.ndarray
    my_output = corr1d.short_filter_correlate1D(test_data, test_filter)  # type: np.ndarray

    assert my_output.dtype == np.float64, 'data type should be np.float64'

    assert comparison.shape == my_output.shape, 'shape is {0}, should be {1}'.format(my_output.shape,
                                                                                     comparison.shape)

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_corr1D_AC_float():
    test_data = FILTERS[0, 35, :].astype(np.float32)
    test_filter = FILTERS[0, 35, :].astype(np.float32)

    comparison = np.correlate(test_data, test_filter, mode='valid')
    my_output = corr1d.short_filter_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float32, 'data type should be np.float32'

    assert comparison.shape == my_output.shape, 'shape is {0}, should be {1}'.format(my_output.shape,
                                                                                     comparison.shape)

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_filter_multidata_1D_A_double():
    test_data = DATA[0, ...].astype(np.float64)
    test_filter = FILTERS[0, 35, :].astype(np.float64)

    comparison = np.zeros((test_data.shape[0], test_data.shape[1] - test_filter.shape[0] + 1),
                          dtype=np.float64)
    for i in range(test_data.shape[0]):
        comparison[i, ...] = np.correlate(test_data[i, :], test_filter)

    my_output = corr1d.single_filter_multiple_data_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float64
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_filter_multidata_1D_A_float():
    test_data = DATA[0, ...].astype(np.float32)
    test_filter = FILTERS[0, 35, :].astype(np.float32)

    comparison = np.zeros((test_data.shape[0], test_data.shape[1] - test_filter.shape[0] + 1),
                          dtype=np.float32)
    for i in range(test_data.shape[0]):
        comparison[i, ...] = np.correlate(test_data[i, :], test_filter)

    my_output = corr1d.single_filter_multiple_data_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float32
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_filter_multidata_1D_B_double():
    test_data = DATA[0, :, :1001].astype(np.float64)
    test_filter = FILTERS[0, 35, :].astype(np.float64)

    comparison = np.zeros((test_data.shape[0], test_data.shape[1] - test_filter.shape[0] + 1),
                          dtype=np.float64)
    for i in range(test_data.shape[0]):
        comparison[i, ...] = np.correlate(test_data[i, :], test_filter)

    my_output = corr1d.single_filter_multiple_data_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float64
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_filter_multidata_1D_B_float():
    test_data = DATA[0, :, :1001].astype(np.float32)
    test_filter = FILTERS[0, 35, :].astype(np.float32)

    comparison = np.zeros((test_data.shape[0], test_data.shape[1] - test_filter.shape[0] + 1),
                          dtype=np.float32)
    for i in range(test_data.shape[0]):
        comparison[i, ...] = np.correlate(test_data[i, :], test_filter)

    my_output = corr1d.single_filter_multiple_data_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float32
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_data_multifilter_1D_A_double():
    test_data = DATA[0, 35, :].astype(np.float64)
    test_filter = FILTERS[0, :, :].astype(np.float64)

    comparison = np.zeros((test_filter.shape[0], test_data.shape[0] - test_filter.shape[1] + 1),
                          dtype=np.float64)
    for i in range(test_filter.shape[0]):
        comparison[i, ...] = np.correlate(test_data, test_filter[i, :])

    my_output = corr1d.multiple_filter_single_data_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float64
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_data_multifilter_1D_A_float():
    test_data = DATA[0, 35, :].astype(np.float32)
    test_filter = FILTERS[0, :, :].astype(np.float32)

    comparison = np.zeros((test_filter.shape[0], test_data.shape[0] - test_filter.shape[1] + 1),
                          dtype=np.float32)
    for i in range(test_filter.shape[0]):
        comparison[i, ...] = np.correlate(test_data, test_filter[i, :])

    my_output = corr1d.multiple_filter_single_data_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float32
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_data_multifilter_1D_B_double():
    test_data = DATA[0, 54, 1337:9999].astype(np.float64)
    test_filter = FILTERS[0, :, :].astype(np.float64)

    comparison = np.zeros((test_filter.shape[0], test_data.shape[0] - test_filter.shape[1] + 1),
                          dtype=np.float64)
    for i in range(test_filter.shape[0]):
        comparison[i, ...] = np.correlate(test_data, test_filter[i, :])

    my_output = corr1d.multiple_filter_single_data_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float64
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_data_multifilter_1D_B_float():
    test_data = DATA[0, 54, 1337:9999].astype(np.float32)
    test_filter = FILTERS[0, :, :].astype(np.float32)

    comparison = np.zeros((test_filter.shape[0], test_data.shape[0] - test_filter.shape[1] + 1),
                          dtype=np.float32)
    for i in range(test_filter.shape[0]):
        comparison[i, ...] = np.correlate(test_data, test_filter[i, :])

    my_output = corr1d.multiple_filter_single_data_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float32
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_multidata_multifilter_1D_A_double():
    test_data = DATA[0, 13:26, :].astype(np.float64)
    test_filter = FILTERS[0, 15:19, :].astype(np.float64)

    comparison = np.zeros((test_data.shape[0], test_filter.shape[0], test_data.shape[1] - test_filter.shape[1] + 1),
                          dtype=np.float64)

    for j in range(test_data.shape[0]):
        for i in range(test_filter.shape[0]):
            comparison[j, i, :] = np.correlate(test_data[j, :], test_filter[i, :])

    my_output = corr1d.multiple_filter_multiple_data_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float64
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_multidata_multifilter_1D_A_float():
    test_data = DATA[0, 13:26, :].astype(np.float32)
    test_filter = FILTERS[0, 15:19, :].astype(np.float32)

    comparison = np.zeros((test_data.shape[0], test_filter.shape[0], test_data.shape[1] - test_filter.shape[1] + 1),
                          dtype=np.float32)

    for j in range(test_data.shape[0]):
        for i in range(test_filter.shape[0]):
            comparison[j, i, :] = np.correlate(test_data[j, :], test_filter[i, :])

    my_output = corr1d.multiple_filter_multiple_data_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float32
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_data_single_filter_accum_A_double():
    test_data = (DATA[0, :, :] / 10.0).astype(np.float64)
    test_filter = (FILTERS[0, :, :] / 10.0).astype(np.float64)

    buffer = np.zeros((test_data.shape[0], test_data.shape[1] - test_filter.shape[1] + 1),
                      dtype=np.float64)

    for j in range(test_data.shape[0]):
        buffer[j, :] = np.correlate(test_data[j, :], test_filter[j, :])
    comparison = np.sum(buffer, axis=0)

    my_output = corr1d.multichan_accum_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float64
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_data_single_filter_accum_A_float():
    test_data = (DATA[0, :, :] / 10.0).astype(np.float32)
    test_filter = (FILTERS[0, :, :] / 10.0).astype(np.float32)

    buffer = np.zeros((test_data.shape[0], test_data.shape[1] - test_filter.shape[1] + 1),
                      dtype=np.float32)

    for j in range(test_data.shape[0]):
        buffer[j, :] = np.correlate(test_data[j, :], test_filter[j, :])
    comparison = np.sum(buffer, axis=0)

    my_output = corr1d.multichan_accum_correlate1D(test_data, test_filter)

    assert my_output.dtype == np.float32
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_data_multifilter_accum_A_double():
    test_data = (DATA[0, :13, :] / 10.0).astype(np.float64)
    test_filter = (FILTERS[:, :13, :] / 10.0).astype(np.float64)

    buffer = np.zeros((test_filter.shape[0], test_data.shape[0], test_data.shape[1] - test_filter.shape[2] + 1),
                      dtype=np.float64)

    for k in range(test_filter.shape[0]):
        for j in range(test_data.shape[0]):
            buffer[k, j, :] = np.correlate(test_data[j, :], test_filter[k, j, :])
    comparison = np.sum(buffer, axis=1)

    my_output = corr1d.batch_filter_multichan_accum_correlate1D(test_data, test_filter)
    print(np.max(np.abs(comparison - my_output)))

    assert my_output.dtype == np.float64
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_single_data_multifilter_accum_A_float():
    test_data = (DATA[0, :13, :] / 10.0).astype(np.float32)
    test_filter = (FILTERS[:, :13, :] / 10.0).astype(np.float32)

    buffer = np.zeros((test_filter.shape[0], test_data.shape[0], test_data.shape[1] - test_filter.shape[2] + 1),
                      dtype=np.float32)

    for k in range(test_filter.shape[0]):
        for j in range(test_data.shape[0]):
            buffer[k, j, :] = np.correlate(test_data[j, :], test_filter[k, j, :])
    comparison = np.sum(buffer, axis=1)

    my_output = corr1d.batch_filter_multichan_accum_correlate1D(test_data, test_filter)
    print(np.max(np.abs(comparison - my_output)))

    assert my_output.dtype == np.float32
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-3, atol=1e-2)


def test_multidata_single_filter_accum_A_double():
    test_data = (DATA[:, :13, :] / 10.0).astype(np.float64)
    test_filter = (FILTERS[0, :13, :] / 10.0).astype(np.float64)

    buffer = np.zeros((test_data.shape[0], test_data.shape[1], test_data.shape[2] - test_filter.shape[1] + 1),
                      dtype=np.float64)

    for k in range(test_data.shape[0]):
        for j in range(test_data.shape[1]):
            buffer[k, j, :] = np.correlate(test_data[k, j, :], test_filter[j, :])
    comparison = np.sum(buffer, axis=1)

    my_output = corr1d.batch_data_multichan_accum_correlate1D(test_data, test_filter)
    print(np.max(np.abs(comparison - my_output)))

    assert my_output.dtype == np.float64
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-2, atol=1e-2)


def test_multidata_single_filter_accum_A_float():
    test_data = (DATA[:, :13, :] / 10.0).astype(np.float32)
    test_filter = (FILTERS[0, :13, :] / 10.0).astype(np.float32)

    buffer = np.zeros((test_data.shape[0], test_data.shape[1], test_data.shape[2] - test_filter.shape[1] + 1),
                      dtype=np.float32)

    for k in range(test_data.shape[0]):
        for j in range(test_data.shape[1]):
            buffer[k, j, :] = np.correlate(test_data[k, j, :], test_filter[j, :])
    comparison = np.sum(buffer, axis=1)

    my_output = corr1d.batch_data_multichan_accum_correlate1D(test_data, test_filter)
    print(np.max(np.abs(comparison - my_output)))

    assert my_output.dtype == np.float32
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-2, atol=1e-2)


def test_multidata_multifilter_accum_A_double():
    test_data = (DATA[:, :13, :] / 10.0).astype(np.float64)
    test_filter = (FILTERS[:, :13, :] / 10.0).astype(np.float64)

    buffer = np.zeros((test_data.shape[0], test_filter.shape[0],
                       test_data.shape[1], test_data.shape[2] - test_filter.shape[2] + 1),
                      dtype=np.float64)

    for k in range(test_data.shape[0]):
        for l in range(test_filter.shape[0]):
            for j in range(test_data.shape[1]):
                buffer[k, l, j, :] = np.correlate(test_data[k, j, :], test_filter[l, j, :])
    comparison = np.sum(buffer, axis=2)

    my_output = corr1d.batch_data_batch_filter_multichan_accum_correlate1D(test_data, test_filter)
    print(np.max(np.abs(comparison - my_output)))

    assert my_output.dtype == np.float64
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-2, atol=1e-2)


def test_multidata_multifilter_accum_A_float():
    test_data = (DATA[:, :13, :] / 10.0).astype(np.float32)
    test_filter = (FILTERS[:, :13, :] / 10.0).astype(np.float32)

    buffer = np.zeros((test_data.shape[0], test_filter.shape[0],
                       test_data.shape[1], test_data.shape[2] - test_filter.shape[2] + 1),
                      dtype=np.float32)

    for k in range(test_data.shape[0]):
        for l in range(test_filter.shape[0]):
            for j in range(test_data.shape[1]):
                buffer[k, l, j, :] = np.correlate(test_data[k, j, :], test_filter[l, j, :])
    comparison = np.sum(buffer, axis=2)

    my_output = corr1d.batch_data_batch_filter_multichan_accum_correlate1D(test_data, test_filter)
    print(np.max(np.abs(comparison - my_output)))

    assert my_output.dtype == np.float32
    assert my_output.shape == comparison.shape

    assert np.allclose(comparison, my_output, rtol=1e-2, atol=1e-2)


def test__single_filter_single_data_channel_correlate1D():
    N_CH = 13
    test_data = (DATA[0, :N_CH, :] / 10.0).astype(np.float64)
    test_filter = (FILTERS[0, :N_CH, :] / 10.0).astype(np.float64)

    buffer = np.zeros((test_data.shape[0], test_data.shape[1] - test_filter.shape[1] + 1),
                      dtype=np.float64)

    for k in range(N_CH):
        buffer[k, :] = np.correlate(test_data[k, :], test_filter[k, :])

    my_output = corr1d.single_filter_single_data_channel_correlate1D(test_data,
                                                                     test_filter)
    print(np.max(np.abs(buffer - my_output)))

    assert my_output.dtype == np.float64
    assert my_output.shape == buffer.shape

    assert np.allclose(buffer, my_output, rtol=1e-2, atol=1e-2)


def test__single_filter_single_data_channel_correlate1D__float():
    N_CH = 13
    test_data = (DATA[0, :N_CH, :] / 10.0).astype(np.float32)
    test_filter = (FILTERS[0, :N_CH, :] / 10.0).astype(np.float32)

    buffer = np.zeros((test_data.shape[0], test_data.shape[1] - test_filter.shape[1] + 1),
                      dtype=np.float32)

    for k in range(N_CH):
        buffer[k, :] = np.correlate(test_data[k, :], test_filter[k, :])

    my_output = corr1d.single_filter_single_data_channel_correlate1D(test_data,
                                                                     test_filter)
    print(np.max(np.abs(buffer - my_output)))

    assert my_output.dtype == np.float32
    assert my_output.shape == buffer.shape

    assert np.allclose(buffer, my_output, rtol=1e-2, atol=1e-2)


def test__single_filter_batch_data_channel_correlate1D():
    N_CH = 13
    test_data = (DATA[:, :N_CH, :] / 10.0).astype(np.float64)
    test_filter = (FILTERS[0, :N_CH, :] / 10.0).astype(np.float64)

    buffer = np.zeros((test_data.shape[0], test_data.shape[1], test_data.shape[2] - test_filter.shape[1] + 1),
                      dtype=np.float64)

    for i in range(test_data.shape[0]):
        for k in range(test_data.shape[1]):
            buffer[i, k, :] = np.correlate(test_data[i, k, :], test_filter[k, :])

    my_output = corr1d.single_filter_batch_data_channel_correlate1D(test_data,
                                                                    test_filter)
    print(np.max(np.abs(buffer - my_output)))

    assert my_output.dtype == np.float64
    assert my_output.shape == buffer.shape

    assert np.allclose(buffer, my_output, rtol=1e-2, atol=1e-2)


def test__single_filter_batch_data_channel_correlate1D__float():
    N_CH = 13
    test_data = (DATA[:, :N_CH, :] / 10.0).astype(np.float32)
    test_filter = (FILTERS[0, :N_CH, :] / 10.0).astype(np.float32)

    buffer = np.zeros((test_data.shape[0], test_data.shape[1], test_data.shape[2] - test_filter.shape[1] + 1),
                      dtype=np.float32)

    for i in range(test_data.shape[0]):
        for k in range(test_data.shape[1]):
            buffer[i, k, :] = np.correlate(test_data[i, k, :], test_filter[k, :])

    my_output = corr1d.single_filter_batch_data_channel_correlate1D(test_data,
                                                                    test_filter)
    print(np.max(np.abs(buffer - my_output)))

    assert my_output.dtype == np.float32
    assert my_output.shape == buffer.shape

    assert np.allclose(buffer, my_output, rtol=1e-2, atol=1e-2)


def test__batch_filter_single_data_channel_correlate1D():
    N_CH = 13
    test_data = (DATA[0, :N_CH, :] / 10.0).astype(np.float64)
    test_filter = (FILTERS[:, :N_CH, :] / 10.0).astype(np.float64)

    buffer = np.zeros((test_filter.shape[0], test_filter.shape[1], test_data.shape[1] - test_filter.shape[2] + 1),
                      dtype=np.float64)

    for i in range(test_filter.shape[0]):
        for k in range(N_CH):
            buffer[i, k, :] = np.correlate(test_data[k, :], test_filter[i, k, :])

    my_output = corr1d.batch_filter_single_data_channel_correlate1D(test_data,
                                                                    test_filter)
    print(np.max(np.abs(buffer - my_output)))

    assert my_output.dtype == np.float64
    assert my_output.shape == buffer.shape

    assert np.allclose(buffer, my_output, rtol=1e-2, atol=1e-2)


def test__batch_filter_single_data_channel_correlate1D__float():
    N_CH = 13
    test_data = (DATA[0, :N_CH, :] / 10.0).astype(np.float32)
    test_filter = (FILTERS[:, :N_CH, :] / 10.0).astype(np.float32)

    buffer = np.zeros((test_filter.shape[0], test_filter.shape[1], test_data.shape[1] - test_filter.shape[2] + 1),
                      dtype=np.float32)

    for i in range(test_filter.shape[0]):
        for k in range(N_CH):
            buffer[i, k, :] = np.correlate(test_data[k, :], test_filter[i, k, :])

    my_output = corr1d.batch_filter_single_data_channel_correlate1D(test_data,
                                                                    test_filter)
    print(np.max(np.abs(buffer - my_output)))

    assert my_output.dtype == np.float32
    assert my_output.shape == buffer.shape

    assert np.allclose(buffer, my_output, rtol=1e-2, atol=1e-2)


def test__batch_filter_batch_data_channel_correlate1D():
    N_CH = 13
    test_data = (DATA[:, :N_CH, :] / 10.0).astype(np.float64)
    test_filter = (FILTERS[:, :N_CH, :] / 10.0).astype(np.float64)

    buffer = np.zeros((test_data.shape[0], test_filter.shape[0], test_filter.shape[1],
                       test_data.shape[2] - test_filter.shape[2] + 1),
                      dtype=np.float64)

    for j in range(test_data.shape[0]):
        for i in range(test_filter.shape[0]):
            for k in range(N_CH):
                buffer[j, i, k, :] = np.correlate(test_data[j, k, :], test_filter[i, k, :])

    my_output = corr1d.batch_filter_batch_data_channel_correlate1D(test_data,
                                                                   test_filter)
    print(np.max(np.abs(buffer - my_output)))

    assert my_output.dtype == np.float64
    assert my_output.shape == buffer.shape

    assert np.allclose(buffer, my_output, rtol=1e-2, atol=1e-2)


def test__batch_filter_batch_data_channel_correlate1D__float():
    N_CH = 13
    test_data = (DATA[:, :N_CH, :] / 10.0).astype(np.float32)
    test_filter = (FILTERS[:, :N_CH, :] / 10.0).astype(np.float32)

    buffer = np.zeros((test_data.shape[0], test_filter.shape[0], test_filter.shape[1],
                       test_data.shape[2] - test_filter.shape[2] + 1),
                      dtype=np.float32)

    for j in range(test_data.shape[0]):
        for i in range(test_filter.shape[0]):
            for k in range(N_CH):
                buffer[j, i, k, :] = np.correlate(test_data[j, k, :], test_filter[i, k, :])

    my_output = corr1d.batch_filter_batch_data_channel_correlate1D(test_data,
                                                                   test_filter)
    print(np.max(np.abs(buffer - my_output)))

    assert my_output.dtype == np.float32
    assert my_output.shape == buffer.shape

    assert np.allclose(buffer, my_output, rtol=1e-2, atol=1e-2)
