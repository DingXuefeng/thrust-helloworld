/*****************************************************************************/
// Author: Xuefeng Ding <dingxf@ihep.ac.cn> @ IHEP-CAS
//
// Date: 2023 April 30
// Version: v1.0
// Description: Thrust hello world
//
// All rights reserved. 2023 copyrighted.
/*****************************************************************************/
#include <thrust/transform_reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/functional.h>
#include <iostream>
#ifdef USE_FLOAT
using real_t = float;
#else
using real_t = double;
#endif

struct op {
    void run() {
        auto sum = thrust::transform_reduce(
            thrust::counting_iterator<int>(0),
            thrust::counting_iterator<int>(100),
            [*this] __device__(const int x) -> real_t
            {
                return x * a + b;
            },
            0., thrust::plus<real_t>());
        std::cout << "result: " << sum << std::endl;
    }
    const real_t a,b;
};

int main() {
    const real_t a(3.),b(1.7113);
    auto sum = thrust::transform_reduce(
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(100),
        [a, b] __device__(const int x) -> real_t
        {
            return x * a + b;
        },
        0., thrust::plus<real_t>());
    std::cout << "result: " << sum << std::endl;
    op x {3.,1.7113};
    x.run();
    return 0;
}