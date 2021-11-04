from fastcorr import corr1d
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

    comparison = np.correlate(test_data, test_filter, mode='valid') # type: np.ndarray
    my_output = corr1d.short_filter_correlate1D(test_data, test_filter) # type: np.ndarray

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

    comparison = np.correlate(test_data, test_filter, mode='valid') # type: np.ndarray
    my_output = corr1d.short_filter_correlate1D(test_data, test_filter) # type: np.ndarray

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
