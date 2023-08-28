#include <sycl/sycl.hpp>
#include "utils/tqdm.h"
#include <complex>

static void paint(sycl::queue &q, sycl::buffer<float, 2> &buf) {
    q.submit([&] (sycl::handler &cgh) {
        sycl::accessor axr{buf, cgh, sycl::write_only, sycl::no_init};
        cgh.parallel_for(axr.get_range(), [=] (sycl::id<2> id) {
            auto c = std::complex{(float)id[0] / axr.get_range()[0], (float)id[1] / axr.get_range()[1]};
            auto z = c;
            int iterations = 0;
            while (z.real() * z.real() + z.imag() * z.imag() < 4 && iterations < 50) {
                z = z * z + c;
                iterations++;
            }
            axr[id] = 1 - iterations * 0.02f;
        });
    }).wait_and_throw();
}

int main() {
    constexpr size_t n = 1024;
    {
        sycl::queue q{sycl::cpu_selector_v};
        sycl::buffer<float, 2> buf{sycl::range<2>{n, n}};
        for (auto _: tqdm("cpu", 100)) {
            paint(q, buf);
            sycl::host_accessor hax{buf, sycl::read_only};
            hax[{0, 0}];
        }
    }
    {
        sycl::queue q{sycl::gpu_selector_v};
        sycl::buffer<float, 2> buf{sycl::range<2>{512, 512}};
        for (auto _: tqdm("gpu", 1000)) {
            paint(q, buf);
            sycl::host_accessor hax{buf, sycl::read_only};
            hax[{0, 0}];
        }
    }
    return 0;
}
