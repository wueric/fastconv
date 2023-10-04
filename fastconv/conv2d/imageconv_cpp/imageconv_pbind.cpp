#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "imageconv.h"


PYBIND11_MODULE(imageconv_cpp, m) {
    m.doc() = "Batched 2D image convolution"; // optional module docstring

    m.def("batch_image_batch_filter_2dconv_same",
          &batch_image_batch_filter_2d_conv_same<double>,
          pybind11::return_value_policy::take_ownership,
          "Function that performs 2D batched 'same' convolution for images with a batched kernel");
    m.def("batch_image_batch_filter_2dconv_same",
            &batch_image_batch_filter_2d_conv_same<float>,
            pybind11::return_value_policy::take_ownership,
            "Function that performs 2D batched 'same' convolution for images with a batched kernel");

m.def("batch_smallfilter_2dconv_same",
            &batch_smallfilter_2dconv_same<double>,
            pybind11::return_value_policy::take_ownership,
            "Function that performs 2D batch 'same' convolution for images using a small kernel");

    m.def("batch_smallfilter_2dconv_same",
            &batch_smallfilter_2dconv_same<float>,
            pybind11::return_value_policy::take_ownership,
            "Function that performs 2D batch 'same' convolution for images using a small kernel");

    m.def("batch_smallfilter_2dconv_shrink",
            &batch_smallfilter_2dconv_shrink<double>,
            pybind11::return_value_policy::take_ownership,
            "Function that performs 2D convolution, assuming that the caller takes care of padding");

    m.def("batch_smallfilter_2dconv_shrink",
            &batch_smallfilter_2dconv_shrink<float>,
            pybind11::return_value_policy::take_ownership,
            "Function that performs 2D convolution, assuming that the caller takes care of padding");
}
