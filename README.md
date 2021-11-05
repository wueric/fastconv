# fastconv

### What's in this package

This package contains optimized parallel 1D and 2D correlation and convolution code.
* `corr1d.*`, which contains functions that perform batched 1D correlation and 1D correlate/accumulate operations.
The functions outperform numpy by 5-15x when the data is large and the filters are short, or when the problems can be parallelized.
* `conv2d.*`, which contains functions that perform batched 2D same and valid convolutions.

The core code is implemented in C++ with hand-optimized SSE/AVX intrinsics, so the performance boost is not portable
(for example, don't expect good performance on M1 Mac).

### Dependencies
* numpy
* pybind11
* C++11 compatible compiler
* x86 CPU with AVX extensions and many cores (Intel preferred, no idea what happens on AMD)

### How much extra performance do you get?

For 1D correlation/convolution, you can expect a 5-15x speedup over naive (possibly looped) application of `np.correlate`.
The speedup is largest when the problems are highly embarrassingly parallel (for example, if the batch size is very large),
or if the number of data time samples is very large (at least 1000). The speedup is negligible (or perhaps a slowdown)
in the single-data-channel single-filter case if both the data and filter are very short (on the order of 150 samples each).

The `np.float32` versions of the functions (invoked automatically if all of the input arrays have dtype `np.float32`) is around 2-4x faster
than the `np.float64` versions.

### Example use cases

#### 1D correlation code
* **Fast parallel high-pass filtering of Hierlemann array raw recorded data.** Every channel of Hierlemann array data 
needs to high-pass filtered with an FIR filter before spike-sorting. This requires convolving chunks of data, where each chunk 
contains ~1000 channels of data, each with ~20,0000-100,000 samples, with a relatively short FIR filter of length ~150.
We can use `corr1d.single_filter_multiple_data_correlate1D`
to quickly (in parallel) filter every channel with the same FIR filter. We can typically filter a 1000x20000 chunk of data (1s of data) in
about 0.4s using this code.
* **Fast correlation of STA timecourses with visual stimulus for fitting LNP or subunit models from white noise**. To fit LNP models from
white noise movies, we need to correlate the stimulus movie matrix with shape `(Height, Width, Frames)` with a 1D timecourse. In this case,
we can reshape the stimulus move matrix to shape `(Height * Width, Frames)` and then do the correlation with `corr1d.single_filter_multiple_data_correlate1D`.
* **Multichannel/multicell template matching for spike-sorting.** Suppose we have a chunk of raw data, with shape `(channels, samples)`,
and candidate template matrix with shape `(n_Candidate_Cells, channels, template_length)`, corresponding to N total cells. 
To do template matching, we want to correlate along the time (last dimension) and sum over the channel dimension, giving an output
matrix of shape `(n_Candidate_Cells, samples - template_length + 1)`. This can be done massively in parallel using the function
`corr1d.batch_filter_multichan_accum_correlate1D`.
* **Multiple sequence detection.** Suppose we have a long 1D data sequence, and we want to detect occurrences of many different known short
subsequences. The data vector has shape `(n_Samples, )`, and the subsequence matrix has shape `(n_subsequences, len_subsequence)`. We can
apply `corr1d.multiple_filter_single_data_correlate1D` to find the subsequences.

#### 2D convolution code
* **Preprocessing visual stimulus frames.** Standard Python image processing libraries do not allow for parallel 2D convolutions.
`conv2d` contains code for both valid and same 2D convolutions. The code performs convolutions in parallel for each image. 
We can filter 18,000 128x256 images of type `np.float32` in about 1s total.

### What this package isn't

This package is **NOT** a drop-in replacement for `np.correlate` or `np.convolve`. **It is highly optimized to target the case
where the length of the filter is VERY SHORT and substanstially shorter than the data**. This package does best when the
data to be filtered is either long in the temporal dimension or contains many channels that can be processed in parallel.

For cases where both the data AND filter are quite long and of comparable size, this package will perform very poorly, and 
you are probably better off doing convolutions/correlations with an FFT-based algorithm.

### Installation instructions

#### Option 1, standalone package
To install as a standalone package, run

```shell script
python setup.py install --force
```

#### Option 2, compile and add to PYTHONPATH
To compile the dependencies in a folder, run

```shell script
python setup.py build_ext --inplace --force
```

and then add the folder to your PYTHONPATH.

### Limited documentation

(see Python files `corr1d/corr1d.py` and `conv2d/conv2d.py` for full documentation, should be self-explanatory)