#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "_fast1dcorr.h"


PYBIND11_MODULE(fastcorr_cpp, m) {
    m.doc() = "Fast correlation for filtering data in 1D"; // optional module docstring

    m.def("batched_filters_batched_data_correlate",
            &batched_filters_batched_data_correlate<double>,
            pybind11::return_value_policy::take_ownership,
            "Function that performs double-precision correlation");

    m.def(
            "batched_filters_batched_data_correlate",
            &batched_filters_batched_data_correlate<float>,
            pybind11::return_value_policy::take_ownership,
            "Function that performs correlation"
            );

    m.def("batched_data_batched_filter_multichan_correlate_accum",
            &batched_data_batched_filter_multichan_correlate_accum<double>,
            pybind11::return_value_policy::take_ownership,
            "Function that performs double-precision multichannel correlate-accumulate");

    m.def("batched_data_batched_filter_multichan_correlate_accum",
            &batched_data_batched_filter_multichan_correlate_accum<float>,
            pybind11::return_value_policy::take_ownership,
            "Function that performs multichannel correlate-accumulate");

}

