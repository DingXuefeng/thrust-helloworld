/*****************************************************************************/
// Author: Xuefeng Ding <dingxf@ihep.ac.cn> @ IHEP-CAS
//
// Date: 2023 April 30
// Version: v1.0
// Description: Thrust hello world
//
// All rights reserved. 2023 copyrighted.
/*****************************************************************************/
#pragma once

__device__ int libB(int x);
struct libA_functor {
    libA_functor(int *a) : d_ptr(a) {}
    __device__ int operator()(int x);
    int *d_ptr;
    // thrust::device_vector<int> d_vec; // do not add this.
};
