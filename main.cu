/*****************************************************************************/
// Author: Xuefeng Ding <dingxf@ihep.ac.cn> @ IHEP-CAS
//
// Date: 2023 April 30
// Version: v1.0
// Description: Thrust hello world
//
// All rights reserved. 2023 copyrighted.
/*****************************************************************************/
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform.h>
#include <chrono>
#ifdef USE_FLOAT
using real_t = float;
#else
using real_t = double;
#endif
struct conv : public thrust::unary_function<int, real_t>
{
    conv(real_t *x, real_t *fx,
         int task_a, int task_b, int task_c,
         int output) : m_x(x), m_fx(fx),
                       m_tasksize{task_a, task_b, task_c},
                       m_taskshift{0, task_a, task_a + task_b},
                       m_output(output) {}
    void update(
        real_t lightYield, real_t sigma)
    {
        m_lightYield = lightYield;
        m_sigmaInv = 1. / sigma;
    }
    __device__
        real_t
        operator()(const int id)
    {
        const int task_id = id / m_output;
        const int output_id = id % m_output;
        const real_t *const x = m_x + m_taskshift[task_id];
        const real_t *const fx = m_fx + m_taskshift[task_id];

        const real_t y = output_id;

        real_t sum = 0;
        for (int i = 0; i < m_tasksize[task_id]; ++i)
        {
            constexpr real_t inv_sqrt_2pi = 0.3989422804014327;
            const real_t tmp = (y - x[i] * m_lightYield) * m_sigmaInv;
            constexpr real_t half = 0.5;
            const real_t detector = inv_sqrt_2pi * exp(-tmp * tmp * half) * m_sigmaInv;
            sum += fx[i] * detector;
        }
        return sum;
    }

    const real_t *const m_x;
    const real_t *const m_fx;
    const int m_tasksize[3];
    const int m_taskshift[3];
    const int m_output;

    real_t m_lightYield;
    real_t m_sigmaInv;
};

class Task
{
public:
    Task(int task_a, int task_b, int task_c,
         int output,
         real_t *x, real_t *fx) : m_task_a(task_a), m_task_b(task_b), m_task_c(task_c),
                                  m_x(task_a + task_b + task_c),
                                  m_fx(task_a + task_b + task_c),
                                  m_output(output),
                                  m_y(output * 3),
                                  op(thrust::raw_pointer_cast(m_x.data()),
                                     thrust::raw_pointer_cast(m_fx.data()),
                                     task_a, task_b, task_c,
                                     output)
    {
        thrust::copy(x, x + task_a + task_b + task_c, m_x.begin());
        thrust::copy(fx, fx + task_a + task_b + task_c, m_fx.begin());
    }
    void run(real_t lightYield, real_t sigma)
    {
        op.update(lightYield, sigma);
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(m_y.size()),
                          m_y.begin(),
                          op);
    }
    void output()
    {
        thrust::host_vector<real_t> y = m_y;
        std::vector<real_t> y_sum(m_output);
        for (int i = 0; i < m_output; ++i)
            for (int j = 0; j < 5; ++j)
                y_sum[i] += y[i + j * m_output];
        for (int i = 0; i < 10; ++i)
            std::cout << i << " " << y_sum[i] << std::endl;
    }

private:
    const int m_task_a, m_task_b, m_task_c;
    const int m_output;
    thrust::device_vector<real_t> m_x;
    thrust::device_vector<real_t> m_y;
    thrust::device_vector<real_t> m_fx;
    conv op;
};

int main()
{
    // prepare input
    constexpr int task_a = 1000, task_b = 200, task_c = 4500;
    constexpr int total = task_a + task_b + task_c;
    constexpr int output = 1000;
    real_t x[total], fx[total];
    for (int i = 0; i < task_a; ++i)
    {
        x[i] = 1.0 * i / task_a;
        fx[i] = 5.3 * exp(-x[i] / 0.3);
    }
    for (int i = 0; i < task_b; ++i)
    {
        x[task_a + i] = 0.2 * i / task_b;
        fx[task_a + i] = 5.3 * exp(-x[task_a + i] / 0.03);
    }
    for (int i = 0; i < task_c; ++i)
    {
        x[task_a + task_b + i] = 1.5 * i / task_c;
        fx[task_a + task_b + i] = 5.3 * x[task_a + task_b + i] * (1.5 - x[task_a + task_b + i]);
    }
    // pass to device
    Task hi(task_a, task_b, task_c, output, x, fx);

    auto wall_t0 = std::chrono::high_resolution_clock::now();
    hi.run(500, 20);
    auto wall_t1 = std::chrono::high_resolution_clock::now();
    auto wall_t = std::chrono::duration<real_t, std::milli>(wall_t1 - wall_t0).count();
    std::cout << __LINE__ << "  " << wall_t << " ms" <<std::endl;

    hi.output();
    return 0;
}