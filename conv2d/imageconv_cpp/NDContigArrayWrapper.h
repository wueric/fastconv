//
// Created by Eric Wu on 11/3/21.
//

#ifndef IMAGECONV_NDCONTIGARRAYWRAPPER_H
#define IMAGECONV_NDCONTIGARRAYWRAPPER_H

template<class T> struct TwoDContigArrayWrapper {
    T *array_ptr;
    const int64_t dim0;
    const int64_t dim1;

    TwoDContigArrayWrapper<T> (T *ptr, const int64_t dim0_, const int64_t dim1_) : array_ptr(ptr), dim0(dim0_), dim1(dim1_) { };
};

#endif //FASTCONV_NDCONTIGARRAYWRAPPER_H
