import os
import visionloader as vl
import bin2py

import numpy as np

ABOUT_STR = 'Generate data for test cases for fast 1D correlations'

TEST_DATASET = '/Volumes/Data/2018-08-07-1/data000'
TEST_DATASET_NAME = 'data000'

SPSORT_PATH = '/Volumes/Analysis/2018-08-07-1/data000'

TEMPLATE_CELL_IDS = [541, 3106, 3607, 3018, 1401]

TESTCASE_PATH = 'testcases'
MULTIDATA_MULTICHAN_FNAME = 'multidata_multichan.npy'
MULTICHAN_FILTERS_FNAME = 'multifilters.npy'

N_SAMPLES_DATA = 20001
DATA_START_SAMPLES = [0, 70000, 1000000, 2000000, 5000000]

if __name__ == '__main__':

    os.makedirs(TESTCASE_PATH, exist_ok=True)
    data_file_path = os.path.join(TESTCASE_PATH, MULTIDATA_MULTICHAN_FNAME)

    with bin2py.PyBinFileReader(TEST_DATASET, is_row_major=True) as pbfr:

        n_channels = pbfr.num_electrodes

        output_npy_buffer = np.zeros((len(DATA_START_SAMPLES), n_channels, N_SAMPLES_DATA), dtype=np.float64)
        for i, offset in enumerate(DATA_START_SAMPLES):
            output_npy_buffer[i,...] = pbfr.get_data(offset, N_SAMPLES_DATA)[1:,:]

    np.save(data_file_path, output_npy_buffer, allow_pickle=True)

    filter_file_path = os.path.join(TESTCASE_PATH, MULTICHAN_FILTERS_FNAME)
    analysis_dset = vl.load_vision_data(SPSORT_PATH, TEST_DATASET_NAME, include_params=True, include_ei=True)
    ei_nchan, ei_nsamples = analysis_dset.get_ei_for_cell(TEMPLATE_CELL_IDS[0]).ei.shape
    ei_buffer = np.zeros((len(TEMPLATE_CELL_IDS), n_channels, ei_nsamples), dtype=np.float64)

    for i, cell_id in enumerate(TEMPLATE_CELL_IDS):
        ei_buffer[i, ...] = analysis_dset.get_ei_for_cell(cell_id).ei

    np.save(filter_file_path, ei_buffer, allow_pickle=True)
