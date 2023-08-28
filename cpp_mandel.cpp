#include "utils/tqdm.h"
#include <complex>
#include <numeric>
#include <vector>
#if __has_include(<tbb/parallel_for.h>)
#include <tbb/parallel_for.h>
#include <tbb/blocked_range2d.h>
#endif
#if __has_include(<experimental/simd>)
#include <experimental/simd>
#endif

static void paint(float *buf, int n) {
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            auto c = std::complex<float>{x / (float)n, y / (float)n};
            auto z = c;
            int iterations = 0;
            while (z.real() * z.real() + z.imag() * z.imag() < 4 && iterations < 50) {
                z = z * z + c;
                iterations++;
            }
            buf[x + y * n] = 1 - iterations * 0.02f;
        }
    }
}

#if __has_include(<experimental/simd>)
static void paint_simd(float *buf, int n) {
    using namespace std::experimental;
    for (int y = 0; y < n; y++) {
        native_simd<float> xx;
        float xxx[native_simd<float>::size()];
        std::iota(xxx, xxx + native_simd<float>::size(), 0);
        xx.copy_from(xxx, element_aligned);
        auto bufyp = buf + y * n;
        native_simd<float> yy = y / (float)n;
        native_simd<float> dx = native_simd<float>::size() / (float)n;
        for (int x = 0; x < n; x += native_simd<float>::size()) {
            native_simd<float> c;
            xx += dx;
            auto uu = xx, vv = yy;
            int iterations = 0;
            native_simd<float> lastiters = 0;
            while (iterations < 50) {
                native_simd_mask<float> mask = xx * xx + yy * yy < 4.0f;
                if (none_of(mask))
                    break;
                auto new_xx = xx * xx - yy * yy;
                auto new_yy = xx * yy * 2.0f;
                xx = new_xx;
                yy = new_yy;
// 由于 experimental/simd 不支持 native_simd_mask<float> 转成 native_simd_mask<int>，只能强行写入隐式转换的 int iterations
                where(mask, lastiters) = iterations;
                iterations++;
            }
            (1 - lastiters * 0.02f).copy_to(bufyp, element_aligned);
        }
    }
}
#endif

static void paint_omp(float *buf, int n) {
    #pragma omp parallel for collapse(2)
    for (int y = 0; y < n; y++) {
        for (int x = 0; x < n; x++) {
            auto c = std::complex<float>{x / (float)n, y / (float)n};
            auto z = c;
            int iterations = 0;
            while (z.real() * z.real() + z.imag() * z.imag() < 4 && iterations < 50) {
                z = z * z + c;
                iterations++;
            }
            buf[x + y * n] = 1 - iterations * 0.02f;
        }
    }
}

#if __has_include(<experimental/simd>)
static void paint_omp_simd(float *buf, int n) {
    using namespace std::experimental;
    #pragma omp parallel for
    for (int y = 0; y < n; y++) {
        native_simd<float> xx;
        float xxx[native_simd<float>::size()];
        std::iota(xxx, xxx + native_simd<float>::size(), 0);
        xx.copy_from(xxx, element_aligned);
        auto bufyp = buf + y * n;
        native_simd<float> yy = y / (float)n;
        native_simd<float> dx = native_simd<float>::size() / (float)n;
        for (int x = 0; x < n; x += native_simd<float>::size()) {
            native_simd<float> c;
            xx += dx;
            auto uu = xx, vv = yy;
            int iterations = 0;
            native_simd<float> lastiters = 0;
            while (iterations < 50) {
                native_simd_mask<float> mask = xx * xx + yy * yy < 4.0f;
                if (none_of(mask))
                    break;
                auto new_xx = xx * xx - yy * yy;
                auto new_yy = xx * yy * 2.0f;
                xx = new_xx;
                yy = new_yy;
                where(mask, lastiters) = iterations;
                iterations++;
            }
            (1 - lastiters * 0.02f).copy_to(bufyp, element_aligned);
        }
    }
}
#endif

#if __has_include(<tbb/parallel_for.h>)
static void paint_tbb(float *buf, int n) {
    tbb::parallel_for(tbb::blocked_range2d<int>(0, n, 0, n), [buf, n] (tbb::blocked_range2d<int> r) {
        for (int y = r.rows().begin(); y < r.rows().end(); y++) {
            for (int x = r.cols().begin(); x < r.cols().end(); x++) {
                auto c = std::complex<float>{x / (float)n, y / (float)n};
                auto z = c;
                int iterations = 0;
                while (z.real() * z.real() + z.imag() * z.imag() < 4 && iterations < 50) {
                    z = z * z + c;
                    iterations++;
                }
                buf[x + y * n] = 1 - iterations * 0.02f;
            }
        }
    });
}

#if __has_include(<experimental/simd>)
static void paint_tbb_simd(float *buf, int n) {
    using namespace std::experimental;
    tbb::parallel_for(0, n, [buf, n] (int y) {
        native_simd<float> xx;
        float xxx[native_simd<float>::size()];
        std::iota(xxx, xxx + native_simd<float>::size(), 0);
        xx.copy_from(xxx, element_aligned);
        auto bufyp = buf + y * n;
        native_simd<float> yy = y / (float)n;
        native_simd<float> dx = native_simd<float>::size() / (float)n;
        for (int x = 0; x < n; x += native_simd<float>::size()) {
            native_simd<float> c;
            xx += dx;
            auto uu = xx, vv = yy;
            int iterations = 0;
            native_simd<float> lastiters = 0;
            while (iterations < 50) {
                native_simd_mask<float> mask = xx * xx + yy * yy < 4.0f;
                if (none_of(mask))
                    break;
                auto new_xx = xx * xx - yy * yy;
                auto new_yy = xx * yy * 2.0f;
                xx = new_xx;
                yy = new_yy;
                where(mask, lastiters) = iterations;
                iterations++;
            }
            (1 - lastiters * 0.02f).copy_to(bufyp, element_aligned);
        }
    });
}
#endif
#endif

int main() {
    constexpr size_t n = 1024;
    {
        std::vector<float> buf(n * n);
        for (auto _: tqdm("serial", 100)) {
            paint(buf.data(), n);
            asm volatile ("" :: "r,m" (buf[0]) : "cc", "memory");
        }
    }
#if __has_include(<experimental/simd>)
    {
        std::vector<float> buf(n * n);
        for (auto _: tqdm("serial+simd", 100)) {
            paint_simd(buf.data(), n);
            asm volatile ("" :: "r,m" (buf[0]) : "cc", "memory");
        }
    }
#endif
    {
        std::vector<float> buf(n * n);
        for (auto _: tqdm("openmp", 100)) {
            paint_omp(buf.data(), n);
            asm volatile ("" :: "r,m" (buf[0]) : "cc", "memory");
        }
    }
#if __has_include(<experimental/simd>)
    {
        std::vector<float> buf(n * n);
        for (auto _: tqdm("openmp+simd", 100)) {
            paint_omp_simd(buf.data(), n);
            asm volatile ("" :: "r,m" (buf[0]) : "cc", "memory");
        }
    }
#endif
#if __has_include(<tbb/parallel_for.h>)
    {
        std::vector<float> buf(n * n);
        for (auto _: tqdm("tbb", 100)) {
            paint_tbb(buf.data(), n);
            asm volatile ("" :: "r,m" (buf[0]) : "cc", "memory");
        }
    }
#if __has_include(<experimental/simd>)
    {
        std::vector<float> buf(n * n);
        for (auto _: tqdm("tbb+simd", 100)) {
            paint_tbb_simd(buf.data(), n);
            asm volatile ("" :: "r,m" (buf[0]) : "cc", "memory");
        }
    }
#endif
#endif
    return 0;
}
