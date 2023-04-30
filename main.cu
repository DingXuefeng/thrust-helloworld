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
#include <nvToolsExt.h>
#include <chrono>
#include <vector>
#ifdef USE_FLOAT
using real_t = float;
#else
using real_t = double;
#endif

struct conv : public thrust::unary_function<int, real_t>
{
    conv(int *nInput,
         real_t **x,
         real_t **fx,
         int output)
        : m_N(nInput),
          m_x(x),
          m_fx(fx),
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

        const real_t my = -1.0 * output_id;

        real_t sum = 0;
        for (int i = 0; i < m_N[task_id]; ++i)
        {
            constexpr real_t inv_sqrt_2pi = 0.3989422804014327;
            const real_t tmp = (m_x[task_id][i] * m_lightYield + my) * m_sigmaInv;
            constexpr real_t mhalf = -0.5;
            const real_t detector = inv_sqrt_2pi * exp(mhalf * tmp * tmp) * m_sigmaInv;
            sum += m_fx[task_id][i] * detector;
        }
        return sum;
    }

    int *const m_N;
    real_t **const m_x;
    real_t **const m_fx;
    const int m_output;

    real_t m_lightYield;
    real_t m_sigmaInv;
};

class Task
{
public:
    Task(const std::vector<std::vector<real_t>> &x,
         const std::vector<std::vector<real_t>> &fx,
         int output) : m_N(x.size()),
                       m_x(x.size()),
                       m_fx(x.size()),
                       m_output(output),
                       m_y(output * x.size()),
                       op(thrust::raw_pointer_cast(m_N.data()),
                          thrust::raw_pointer_cast(m_x.data()),
                          thrust::raw_pointer_cast(m_fx.data()),
                          output)
    {
        for (int i = 0; i < x.size(); ++i)
        {
            m_N[i] = x[i].size();
            m_x_store.emplace_back(x[i]);
            m_x[i] = thrust::raw_pointer_cast(m_x_store[i].data());
            m_fx_store.emplace_back(fx[i]);
            m_fx[i] = thrust::raw_pointer_cast(m_fx_store[i].data());
        }
    }
    void run(real_t lightYield, real_t sigma)
    {
        op.update(lightYield, sigma);
        nvtxRangePushA("transform");
        thrust::transform(thrust::counting_iterator<int>(0),
                          thrust::counting_iterator<int>(m_y.size()),
                          m_y.begin(),
                          op);
        nvtxRangePop();
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
    const int m_output;
    thrust::device_vector<int> m_N;
    std::vector<thrust::device_vector<real_t>> m_x_store;
    std::vector<thrust::device_vector<real_t>> m_fx_store;
    thrust::device_vector<real_t *> m_x;
    thrust::device_vector<real_t *> m_fx;
    thrust::device_vector<real_t> m_y;
    conv op;
};

int main()
{
    // prepare input
    constexpr int task_a = 1000, task_b = 200, task_c = 4500;
    std::vector<std::vector<real_t>> x(3), fx(3);
    constexpr int output = 19315;
    for (int i = 0; i < task_a; ++i)
    {
        x[0].push_back(1.0 * i / task_a);
        fx[0].push_back(5.3 * exp(-x[0][i] / 0.3));
    }
    for (int i = 0; i < task_b; ++i)
    {
        x[1].push_back(0.2 * i / task_b);
        fx[1].push_back(5.3 * exp(-x[1][i] / 0.03));
    }
    for (int i = 0; i < task_c; ++i)
    {
        x[2].push_back(1.5 * i / task_c);
        fx[2].push_back(5.3 * x[2][i] * (1.5 - x[2][i]));
    }
    // pass to device
    Task hi(x, fx, output);

    auto wall_t0 = std::chrono::high_resolution_clock::now();
    hi.run(500 * 19315 / 1000, 20);
    auto wall_t1 = std::chrono::high_resolution_clock::now();
    auto wall_t = std::chrono::duration<real_t, std::milli>(wall_t1 - wall_t0).count();
    std::cout << __LINE__ << "  " << wall_t << " ms" << std::endl;

    hi.output();
    return 0;
}
