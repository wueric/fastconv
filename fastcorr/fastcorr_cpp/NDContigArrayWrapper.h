//
// Created by Eric Wu on 11/3/21.
//

#ifndef FASTCONV_NDCONTIGARRAYWRAPPER_H
#define FASTCONV_NDCONTIGARRAYWRAPPER_H

template<class T> struct OneDContigArrayWrapper {
    T *array_ptr;
    const int64_t dim0;

    OneDContigArrayWrapper<T> (T *ptr, const int64_t dim0_) : array_ptr(ptr), dim0(dim0_) { };
};

template<class T> struct TwoDContigArrayWrapper {
    T *array_ptr;
    const int64_t dim1;
    const int64_t dim0;

    TwoDContigArrayWrapper<T> (T *ptr, const int64_t dim1_, const int64_t dim0_) : array_ptr(ptr), dim1(dim1_), dim0(dim0_) { };
};

#endif //FASTCONV_NDCONTIGARRAYWRAPPER_H
