//
// Created by Eric Wu on 11/3/21.
//

#ifndef FASTCONV_NDCONTIGARRAYWRAPPER_H
#define FASTCONV_NDCONTIGARRAYWRAPPER_H

template<class T> struct OneDContigArrayWrapper {
    T *array_ptr;
    const size_t dim0;
};

template<class T> struct TwoDContigArrayWrapper {
    T *array_ptr;
    const size_t dim1;
    const size_t dim0;
};

#endif //FASTCONV_NDCONTIGARRAYWRAPPER_H
