from fastcorr import corr1d
import numpy as np

import os

import timeit

TESTCASE_PATH = 'testcases'
MULTIDATA_MULTICHAN_FNAME = 'multidata_multichan.npy'
MULTICHAN_FILTERS_FNAME = 'multifilters.npy'

FILTER_PATH = os.path.join(TESTCASE_PATH, MULTICHAN_FILTERS_FNAME)
DATA_PATH = os.path.join(TESTCASE_PATH, MULTIDATA_MULTICHAN_FNAME)

FILTERS = np.load(FILTER_PATH)
DATA = np.load(DATA_PATH)

if __name__ == '__main__':

    print("Benchmarking 1D correlation")
    data_onechan = DATA[0, 54, :].astype(np.float64).ascontiguousarray()
    filter_onechan = FILTERS[0, 54, :].astype(np.float64).ascontiguousarray()

    t = timeit.Timer(lambda: np.correlate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    t = timeit.Timer(lambda: corr1d.short_filter_correlate1D(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

