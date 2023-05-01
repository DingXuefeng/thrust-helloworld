/*****************************************************************************/
// Author: Xuefeng Ding <dingxf@ihep.ac.cn> @ IHEP-CAS
//
// Date: 2023 April 30
// Version: v1.0
// Description: Thrust hello world
//
// All rights reserved. 2023 copyrighted.
/*****************************************************************************/
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/host_vector.h>
#include <iostream>
#include "libB.h"

__device__ int libA(int x) {
    return libB(x)+1;
}

void run()
{
    int a_init[]{1, 2, 3};
    thrust::device_vector<int> dev_a(a_init, a_init + 3);
    libA_functor op;

    // thrust::transform(dev_a.begin(), dev_a.end(), dev_a.begin(), libA); // not working
    thrust::transform(dev_a.begin(), dev_a.end(), dev_a.begin(), op); // ok
    // thrust::transform(dev_a.begin(), dev_a.end(), dev_a.begin(), libA(thrust::placeholders::_1)); // not working
    // thrust::transform(dev_a.begin(), dev_a.end(), dev_a.begin(), thrust::placeholders::_1+3); // ok. limited op

    thrust::host_vector<int> a = dev_a;
    for (auto x : a)
        std::cout << x << std ::endl;
}