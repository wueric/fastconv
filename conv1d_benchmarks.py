from fastconv import corr1d
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

def benchmark_single_filter_multi_data(multi_data, single_filter):

    data_len, filter_len = multi_data.shape[1], single_filter.shape[0]
    output_len = data_len - filter_len + 1

    output = np.zeros((multi_data.shape[0], output_len), dtype=multi_data.dtype)
    for i in range(multi_data.shape[0]):
        output[i,:] = np.correlate(multi_data[i,:], single_filter)

    return output

def benchmark_single_accumulate(multi_data, multi_filter):

    n_chan, data_len = multi_data.shape
    _, filter_len = multi_filter.shape

    output_len = data_len - filter_len + 1

    temp = np.zeros((output_len, ), dtype=multi_data.dtype)
    for i in range(n_chan):
        temp += np.correlate(multi_data[i,:], multi_filter[i,:])
    return temp

def benchmark_multiple_data_multiple_filter_accumulate(multi_data, multi_filter):
    batch_data, n_chan, data_len = multi_data.shape
    batch_filter, n_chan, filter_len = multi_filter.shape

    output_len = data_len - filter_len + 1
    output = np.zeros((batch_data, batch_filter, output_len), dtype=multi_data.dtype)

    for i in range(batch_data):
        for j in range(batch_filter):
            for k in range(n_chan):
                output[i,j,:] += np.correlate(multi_data[i,k,:], multi_filter[j,k,:])
    return output


if __name__ == '__main__':

    ########### 1D cor ###################################################
    data_onechan = np.ascontiguousarray(DATA[0, 54, :].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float64))
    print("Float64 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.short_filter_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: np.correlate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))


    data_onechan = np.ascontiguousarray(DATA[0, 54, :151].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float64))
    print("Float64 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.short_filter_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: np.correlate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))


    data_onechan = np.ascontiguousarray(DATA[0, 54, :1000].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float64))
    print("Float64 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.short_filter_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: np.correlate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### float32
    data_onechan = np.ascontiguousarray(DATA[0, 54, :].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float32))
    print("Float32 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.short_filter_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: np.correlate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))


    data_onechan = np.ascontiguousarray(DATA[0, 54, :151].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float32))
    print("Float32 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.short_filter_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: np.correlate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))


    print('\n\n\n')

    ########## Single Filter Multiple Data 1D corr #######################
    data_onechan = np.ascontiguousarray(DATA[0, :8, :].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float64))
    print("Float64 batched data single filter 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.single_filter_multiple_data_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_filter_multi_data(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[0, :64, :].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float64))
    print("Float64 batched data single filter 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.single_filter_multiple_data_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_filter_multi_data(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Biggest
    data_onechan = np.ascontiguousarray(DATA[0, :, :].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float64))
    print("Float64 batched data single filter 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.single_filter_multiple_data_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_filter_multi_data(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))


    ### Now float32
    data_onechan = np.ascontiguousarray(DATA[0, :8, :].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float32))
    print("Float32 batched data single filter 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.single_filter_multiple_data_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_filter_multi_data(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[0, :64, :].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float32))
    print("Float32 batched data single filter 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.single_filter_multiple_data_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_filter_multi_data(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Biggest
    data_onechan = np.ascontiguousarray(DATA[0, :, :].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, 54, :].astype(np.float32))
    print("Float32 batched data single filter 1D correlation, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.single_filter_multiple_data_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_filter_multi_data(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    print('\n\n\n')

    ########## Accumulate (single filter set, single data set)  #######################
    data_onechan = np.ascontiguousarray(DATA[0, :8, :].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :8, :].astype(np.float64))
    print("Float64 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[0, :64, :].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :64, :].astype(np.float64))
    print("Float64 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Biggest
    data_onechan = np.ascontiguousarray(DATA[0, :, :].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :, :].astype(np.float64))
    print("Float64 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))


    data_onechan = np.ascontiguousarray(DATA[0, :8, :151].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :8, :151].astype(np.float64))
    print("Float64 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[0, :64, :151].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :64, :151].astype(np.float64))
    print("Float64 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Biggest
    data_onechan = np.ascontiguousarray(DATA[0, :, :151].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :, :151].astype(np.float64))
    print("Float64 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    data_onechan = np.ascontiguousarray(DATA[0, :8, :1000].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :8, :].astype(np.float64))
    print("Float64 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[0, :64, :1000].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :64, :].astype(np.float64))
    print("Float64 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Biggest
    data_onechan = np.ascontiguousarray(DATA[0, :, :1000].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :, :].astype(np.float64))
    print("Float64 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    #float32
    data_onechan = np.ascontiguousarray(DATA[0, :8, :].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :8, :].astype(np.float32))
    print("Float32 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[0, :64, :].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :64, :].astype(np.float32))
    print("Float32 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Biggest
    data_onechan = np.ascontiguousarray(DATA[0, :, :].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :, :].astype(np.float32))
    print("Float32 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))


    data_onechan = np.ascontiguousarray(DATA[0, :8, :151].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :8, :151].astype(np.float32))
    print("Float32 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[0, :64, :151].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :64, :151].astype(np.float32))
    print("Float32 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Biggest
    data_onechan = np.ascontiguousarray(DATA[0, :, :151].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :, :151].astype(np.float32))
    print("Float32 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    data_onechan = np.ascontiguousarray(DATA[0, :8, :1000].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :8, :].astype(np.float32))
    print("Float32 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[0, :64, :1000].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :64, :].astype(np.float32))
    print("Float32 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Biggest
    data_onechan = np.ascontiguousarray(DATA[0, :, :1000].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[0, :, :].astype(np.float32))
    print("Float32 single filter single data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_single_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))



    print('\n\n\n\n\n')

    #### Multidata multifilter accumulate ##############################################3
    data_onechan = np.ascontiguousarray(DATA[:, :8, :].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[:, :8, :].astype(np.float64))
    print("Float64 multiple filter multiple data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.batch_data_batch_filter_multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_multiple_data_multiple_filter_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[:, :64, :].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[:, :64, :].astype(np.float64))
    print("Float64 multiple filter multiple data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.batch_data_batch_filter_multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_multiple_data_multiple_filter_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    # small
    data_onechan = np.ascontiguousarray(DATA[:, :8, :151].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[:, :8, :].astype(np.float64))
    print("Float64 multiple filter multiple data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.batch_data_batch_filter_multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_multiple_data_multiple_filter_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[:, :64, :151].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[:, :64, :].astype(np.float64))
    print("Float64 multiple filter multiple data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.batch_data_batch_filter_multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_multiple_data_multiple_filter_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Biggest
    data_onechan = np.ascontiguousarray(DATA[:, :, :151].astype(np.float64))
    filter_onechan = np.ascontiguousarray(FILTERS[:, :, :].astype(np.float64))
    print("Float64 multiple filter multiple data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.batch_data_batch_filter_multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_multiple_data_multiple_filter_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ##### float32
    data_onechan = np.ascontiguousarray(DATA[:, :8, :].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[:, :8, :].astype(np.float32))
    print("Float32 multiple filter multiple data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.batch_data_batch_filter_multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_multiple_data_multiple_filter_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[:, :64, :].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[:, :64, :].astype(np.float32))
    print("Float32 multiple filter multiple data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.batch_data_batch_filter_multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_multiple_data_multiple_filter_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    # small
    data_onechan = np.ascontiguousarray(DATA[:, :8, :151].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[:, :8, :].astype(np.float32))
    print("Float32 multiple filter multiple data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.batch_data_batch_filter_multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_multiple_data_multiple_filter_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Bigger
    data_onechan = np.ascontiguousarray(DATA[:, :64, :151].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[:, :64, :].astype(np.float32))
    print("Float32 multiple filter multiple data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.batch_data_batch_filter_multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_multiple_data_multiple_filter_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))

    ### Biggest
    data_onechan = np.ascontiguousarray(DATA[:, :, :151].astype(np.float32))
    filter_onechan = np.ascontiguousarray(FILTERS[:, :, :].astype(np.float32))
    print("Float32 multiple filter multiple data correlate/accumulate, data shape {0}, filter shape {1}".format(data_onechan.shape, filter_onechan.shape))

    t = timeit.Timer(lambda: corr1d.batch_data_batch_filter_multichan_accum_correlate1D(data_onechan, filter_onechan))
    print("conv1d: ", t.timeit(10))

    t = timeit.Timer(lambda: benchmark_multiple_data_multiple_filter_accumulate(data_onechan, filter_onechan))
    print("np.correlate: ", t.timeit(10))
