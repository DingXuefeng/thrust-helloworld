/*****************************************************************************/
// Author: Xuefeng Ding <dingxf@ihep.ac.cn> @ IHEP-CAS
//
// Date: 2023 April 30
// Version: v1.0
// Description: Thrust hello world
//
// All rights reserved. 2023 copyrighted.
/*****************************************************************************/
#include "libB.h"

__device__ int libB(int x) {
    return x+1;
}
__device__ int libA_functor::operator()(int x) {
    return libB(x)+d_ptr[0]+d_ptr[1]+d_ptr[2];
}