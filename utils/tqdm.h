#pragma once

#include <chrono>
#include <cstdio>

struct tqdm {
    using Clock = std::chrono::high_resolution_clock;

    const char *name;
    int times;
    Clock::time_point t0;

    tqdm(const char *name, int times) noexcept
        : name(name)
        , times(times)
        , t0(Clock::now())
    {}

    tqdm(tqdm &&) = delete;

    struct iterator {
        using value_type = int;

        int counter;

        int operator*() const {
            return counter;
        }

        iterator &operator++() {
            ++counter;
            return *this;
        }

        iterator operator++(int) {
            iterator tmp = *this;
            ++tmp;
            return tmp;
        }

        bool operator!=(iterator const &that) const {
            return counter != that.counter;
        }

        bool operator==(iterator const &that) const {
            return !(*this != that);
        }
    };

    iterator begin() const {
        return iterator{0};
    }

    iterator end() const {
        return iterator{times};
    }

    ~tqdm() noexcept {
        auto t1 = Clock::now();
        double secs = std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0).count();
        printf("%s: %d 次 %f 秒 %f 次每秒\n", name, times, secs, times / secs);
    }
};
