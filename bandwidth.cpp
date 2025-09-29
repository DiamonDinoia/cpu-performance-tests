#include <omp.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <xsimd/xsimd.hpp>

namespace {

using batch_type = xsimd::batch<double>;

struct Options {
    std::size_t target_bytes;
    int         threads;
    std::size_t iterations;
};

struct ThreadData {
    std::vector<double, xsimd::aligned_allocator<double>> source;
    std::vector<double, xsimd::aligned_allocator<double>> write_target;
    std::vector<double, xsimd::aligned_allocator<double>> copy_target;
    std::vector<double, xsimd::aligned_allocator<double>> flush_buffer;
};

constexpr std::size_t simd_width() { return batch_type::size; }

std::string_view      trim(std::string_view value) {
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.front()))) {
        value.remove_prefix(1);
    }
    while (!value.empty() &&
           std::isspace(static_cast<unsigned char>(value.back()))) {
        value.remove_suffix(1);
    }
    return value;
}

std::size_t parse_size_string(std::string_view value) {
    value = trim(value);
    if (value.empty()) {
        throw std::invalid_argument("Size argument must not be empty");
    }

    std::size_t numeric_end = 0;
    while (numeric_end < value.size() &&
           std::isdigit(static_cast<unsigned char>(value[numeric_end]))) {
        ++numeric_end;
    }

    if (numeric_end == 0) {
        throw std::invalid_argument("Size argument must start with digits: " +
                                    std::string(value));
    }

    std::string number_part(value.substr(0, numeric_end));
    std::string unit_part(value.substr(numeric_end));

    std::size_t numeric_value = 0;
    try {
        numeric_value = std::stoull(number_part);
    } catch (const std::exception &) {
        throw std::invalid_argument("Invalid numeric value for size: " +
                                    number_part);
    }

    std::string normalized_unit;
    normalized_unit.reserve(unit_part.size());
    for (char ch : unit_part) {
        if (!std::isspace(static_cast<unsigned char>(ch))) {
            normalized_unit.push_back(static_cast<char>(
                std::toupper(static_cast<unsigned char>(ch))));
        }
    }

    if (normalized_unit.empty()) { normalized_unit = "MB"; }

    std::size_t multiplier = 0;
    if (normalized_unit == "KB" || normalized_unit == "K") {
        multiplier = 1024ULL;
    } else if (normalized_unit == "MB" || normalized_unit == "M") {
        multiplier = 1024ULL * 1024ULL;
    } else if (normalized_unit == "GB" || normalized_unit == "G") {
        multiplier = 1024ULL * 1024ULL * 1024ULL;
    } else if (normalized_unit == "TB" || normalized_unit == "T") {
        multiplier = 1024ULL * 1024ULL * 1024ULL * 1024ULL;
    } else {
        throw std::invalid_argument("Unsupported size unit: " +
                                    normalized_unit);
    }

    if (numeric_value > std::numeric_limits<std::size_t>::max() / multiplier) {
        throw std::invalid_argument("Size argument is too large");
    }

    return numeric_value * multiplier;
}

int parse_positive_int(std::string_view value, std::string_view name) {
    value = trim(value);
    if (value.empty()) {
        throw std::invalid_argument(std::string(name) +
                                    " value must not be empty");
    }

    for (char ch : value) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::invalid_argument(std::string(name) +
                                        " must be a positive integer");
        }
    }

    int parsed = 0;
    try {
        parsed = std::stoi(std::string(value));
    } catch (const std::exception &) {
        throw std::invalid_argument("Invalid value for " + std::string(name) +
                                    ": " + std::string(value));
    }

    if (parsed <= 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }

    return parsed;
}

std::size_t parse_positive_size_t(std::string_view value,
                                  std::string_view name) {
    value = trim(value);
    if (value.empty()) {
        throw std::invalid_argument(std::string(name) +
                                    " value must not be empty");
    }

    for (char ch : value) {
        if (!std::isdigit(static_cast<unsigned char>(ch))) {
            throw std::invalid_argument(std::string(name) +
                                        " must be a positive integer");
        }
    }

    std::size_t parsed = 0;
    try {
        parsed = std::stoull(std::string(value));
    } catch (const std::exception &) {
        throw std::invalid_argument("Invalid value for " + std::string(name) +
                                    ": " + std::string(value));
    }

    if (parsed == 0) {
        throw std::invalid_argument(std::string(name) + " must be positive");
    }

    return parsed;
}

Options parse_arguments(int argc, char **argv) {
    constexpr std::size_t default_target_bytes = 256ULL * 1024ULL * 1024ULL;
    constexpr std::size_t default_iterations   = 5;

    Options               options{default_target_bytes, omp_get_max_threads(),
                    default_iterations};

    for (int i = 1; i < argc; ++i) {
        std::string_view arg(argv[i]);
        if (arg == "--help" || arg == "-h") { continue; }

        const auto expect_value =
            [&](std::string_view name) -> std::string_view {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for option " +
                                            std::string(name));
            }
            ++i;
            return argv[i];
        };

        if (arg == "--size") {
            options.target_bytes = parse_size_string(expect_value("--size"));
            continue;
        }
        if (arg.rfind("--size=", 0) == 0) {
            std::string_view value =
                arg.substr(std::string_view("--size=").size());
            if (value.empty()) {
                throw std::invalid_argument("Missing value for option --size");
            }
            options.target_bytes = parse_size_string(value);
            continue;
        }

        if (arg == "--threads") {
            options.threads =
                parse_positive_int(expect_value("--threads"), "threads");
            continue;
        }
        if (arg.rfind("--threads=", 0) == 0) {
            std::string_view value =
                arg.substr(std::string_view("--threads=").size());
            if (value.empty()) {
                throw std::invalid_argument(
                    "Missing value for option --threads");
            }
            options.threads = parse_positive_int(value, "threads");
            continue;
        }

        if (arg == "--iterations") {
            options.iterations = parse_positive_size_t(
                expect_value("--iterations"), "iterations");
            continue;
        }
        if (arg.rfind("--iterations=", 0) == 0) {
            std::string_view value =
                arg.substr(std::string_view("--iterations=").size());
            if (value.empty()) {
                throw std::invalid_argument(
                    "Missing value for option --iterations");
            }
            options.iterations = parse_positive_size_t(value, "iterations");
            continue;
        }

        throw std::invalid_argument("Unrecognized option: " + std::string(arg));
    }

    return options;
}

std::string format_megabytes(std::size_t bytes) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2)
        << (static_cast<double>(bytes) / (1024.0 * 1024.0)) << " MB";
    return oss.str();
}

// ---------- SINK to prevent DCE without volatile ----------

#if defined(__clang__) || defined(__GNUG__)
[[gnu::noinline]] static void sink(batch_type v) {
    asm volatile("" : : "x"(v));
}
#else
// Fallback: do a trivial reduction (kept minimal) if inline asm isn't
// supported.
static void sink(batch_type v) { (void)xsimd::hadd(v); }
#endif

// ----------------------------------------------------------

void flush_caches_for_threads(std::vector<ThreadData> &threads,
                              std::vector<double>     &flush_sums) {
#pragma omp parallel
    {
        const int tid       = omp_get_thread_num();
        auto     &buffer    = threads[tid].flush_buffer;
        double    local_sum = 0.0;
        for (double &value : buffer) {
            value = value * 1.0000001 + 1.0;
            local_sum += value;
        }
        flush_sums[tid] = local_sum;
    }
}

[[gnu::noinline]]
void aligned_read_kernel(const std::vector<ThreadData> &threads,
                         std::size_t                    elements_per_thread) {
    const std::size_t width = simd_width();

#pragma omp parallel
    {
        const int     tid  = omp_get_thread_num();
        const double *data = threads[tid].source.data();
        asm volatile("" ::: "memory");
        for (std::size_t index = 0; index + width <= elements_per_thread;
             index += width) {
            auto v = batch_type::load_aligned(data + index);
            sink(v);
        }
        asm volatile("" ::: "memory");
    }
}

[[gnu::noinline]]
void unaligned_read_kernel(const std::vector<ThreadData> &threads,
                           std::size_t                    elements_per_thread) {
    const std::size_t width = simd_width();

#pragma omp parallel
    {
        const int     tid  = omp_get_thread_num();
        const double *data = threads[tid].source.data() + 1;  // misalign
        asm volatile("" ::: "memory");
        for (std::size_t index = 0; index + width <= elements_per_thread;
             index += width) {
            auto v = batch_type::load_unaligned(data + index);
            sink(v);
        }
        asm volatile("" ::: "memory");
    }
}

[[gnu::noinline]]
void aligned_write_kernel(std::vector<ThreadData> &threads,
                          std::size_t elements_per_thread, double value) {
    const std::size_t width = simd_width();
    const batch_type  value_batch(value);

#pragma omp parallel
    {
        const int tid  = omp_get_thread_num();
        double   *data = threads[tid].write_target.data();
        asm volatile("" ::: "memory");
        for (std::size_t index = 0; index + width <= elements_per_thread;
             index += width) {
            value_batch.store_aligned(data + index);
        }
        asm volatile("" ::: "memory");
    }
}

[[gnu::noinline]]
void unaligned_write_kernel(std::vector<ThreadData> &threads,
                            std::size_t elements_per_thread, double value) {
    const std::size_t width = simd_width();
    const batch_type  value_batch(value);

#pragma omp parallel
    {
        const int tid  = omp_get_thread_num();
        double   *data = threads[tid].write_target.data() + 1;  // misalign
        asm volatile("" ::: "memory");
        for (std::size_t index = 0; index + width <= elements_per_thread;
             index += width) {
            value_batch.store_unaligned(data + index);
        }
        asm volatile("" ::: "memory");
    }
}

[[gnu::noinline]]
void aligned_copy_kernel(std::vector<ThreadData> &threads,
                         std::size_t              elements_per_thread) {
    const std::size_t width = simd_width();

#pragma omp parallel
    {
        const int     tid = omp_get_thread_num();
        double       *dst = threads[tid].copy_target.data();
        const double *src = threads[tid].source.data();

        asm volatile("" ::: "memory");
        for (std::size_t index = 0; index + width <= elements_per_thread;
             index += width) {
            batch_type v = batch_type::load_aligned(src + index);
            v.store_aligned(dst + index);
        }
        asm volatile("" ::: "memory");
    }
}

[[gnu::noinline]]
void unaligned_copy_kernel(std::vector<ThreadData> &threads,
                           std::size_t              elements_per_thread) {
    const std::size_t width = simd_width();

#pragma omp parallel
    {
        const int     tid = omp_get_thread_num();
        double       *dst = threads[tid].copy_target.data() + 1;  // misalign
        const double *src = threads[tid].source.data() + 1;       // misalign

        asm volatile("" ::: "memory");
        for (std::size_t index = 0; index + width <= elements_per_thread;
             index += width) {
            batch_type v = batch_type::load_unaligned(src + index);
            v.store_unaligned(dst + index);
        }
        asm volatile("" ::: "memory");
    }
}

template <typename FlushFunc, typename Func, typename FinalizeFunc>
void run_benchmark(const std::string &name, std::size_t bytes_processed,
                   std::size_t iterations, FlushFunc &&flush, Func &&func,
                   FinalizeFunc &&finalize) {
    double total_seconds = 0.0;
    double best_seconds  = std::numeric_limits<double>::infinity();

    for (std::size_t iter = 0; iter < iterations; ++iter) {
        flush();
        const auto start = std::chrono::high_resolution_clock::now();
        func();
        const auto stop = std::chrono::high_resolution_clock::now();
        finalize();
        const double seconds =
            std::chrono::duration<double>(stop - start).count();
        total_seconds += seconds;
        best_seconds = std::min(best_seconds, seconds);
    }

    const double avg_seconds  = total_seconds / static_cast<double>(iterations);
    const double bytes_per_mb = 1024.0 * 1024.0;
    const double avg_bandwidth_mb =
        bytes_processed / avg_seconds / bytes_per_mb;
    const double best_bandwidth_mb =
        bytes_processed / best_seconds / bytes_per_mb;

    std::cout << name << ": " << avg_bandwidth_mb << " MB/s (avg "
              << avg_seconds * 1'000.0 << " ms over " << iterations
              << " runs, best " << best_bandwidth_mb << " MB/s)" << std::endl;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc > 1) {
        for (int i = 1; i < argc; ++i) {
            std::string_view arg(argv[i]);
            if (arg == "--help" || arg == "-h") {
                std::cout << "Usage: " << argv[0]
                          << " [--size <value[KB|MB|GB|TB]>] [--threads "
                             "<count>] [--iterations <count>]"
                          << std::endl;
                return 0;
            }
        }
    }

    Options options{};
    try {
        options = parse_arguments(argc, argv);
    } catch (const std::invalid_argument &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }

    omp_set_num_threads(options.threads);
    const int         thread_count    = omp_get_max_threads();
    const std::size_t width           = simd_width();

    std::size_t       requested_bytes = options.target_bytes;
    if (requested_bytes == 0) {
        requested_bytes =
            static_cast<std::size_t>(thread_count) * width * sizeof(double);
    }

    std::size_t requested_elements =
        (requested_bytes + sizeof(double) - 1) / sizeof(double);
    std::size_t elements_per_thread =
        (requested_elements + static_cast<std::size_t>(thread_count) - 1) /
        static_cast<std::size_t>(thread_count);
    if (elements_per_thread < width) { elements_per_thread = width; }

    const std::size_t total_elements =
        elements_per_thread * static_cast<std::size_t>(thread_count);
    const std::size_t per_thread_bytes = elements_per_thread * sizeof(double);
    const std::size_t total_bytes      = total_elements * sizeof(double);

    const std::size_t flush_elements_per_thread = std::max<std::size_t>(
        elements_per_thread, 8ULL * 1024ULL * 1024ULL / sizeof(double));

    std::vector<ThreadData> thread_data(static_cast<std::size_t>(thread_count));
#pragma omp parallel
    {
        const auto tid  = omp_get_thread_num();
        auto      &data = thread_data[tid];
        data.source.resize(elements_per_thread + width);
        data.write_target.resize(elements_per_thread + width);
        data.copy_target.resize(elements_per_thread + width);
        data.flush_buffer.resize(flush_elements_per_thread);

        std::iota(data.source.begin(), data.source.end(),
                  static_cast<double>(tid));
        std::fill(data.write_target.begin(), data.write_target.end(), 0.0);
        std::fill(data.copy_target.begin(), data.copy_target.end(), 0.0);
        std::iota(data.flush_buffer.begin(), data.flush_buffer.end(), 1.0);
    }

    // bytes actually processed (only full SIMD blocks)
    const std::size_t full_blocks_per_thread =
        (elements_per_thread / width) * width;
    const std::size_t processed_bytes_per_thread =
        full_blocks_per_thread * sizeof(double);
    const std::size_t processed_bytes_all_threads =
        processed_bytes_per_thread * static_cast<std::size_t>(thread_count);
    const std::size_t processed_copy_bytes_all_threads =
        processed_bytes_all_threads * 2;

    std::cout << "Requested size: " << format_megabytes(options.target_bytes)
              << " (" << options.target_bytes << " bytes)" << std::endl;
    if (total_bytes != options.target_bytes) {
        std::cout << "Actual benchmark size: " << format_megabytes(total_bytes)
                  << " (" << total_bytes << " bytes)" << std::endl;
    }
    std::cout << "Per-thread size: " << format_megabytes(per_thread_bytes)
              << " (" << per_thread_bytes << " bytes)" << std::endl;
    std::cout << "OpenMP threads: " << thread_count;
    if (thread_count != options.threads) {
        std::cout << " (requested " << options.threads << ")";
    }
    std::cout << std::endl;
    std::cout << "SIMD width: " << width << " doubles" << std::endl;
    std::cout << "Iterations per test: " << options.iterations << std::endl;

    std::vector<double> flush_sums(static_cast<std::size_t>(thread_count), 0.0);
    volatile double     flush_sink = 0.0;

    auto                flush      = [&] {
        flush_caches_for_threads(thread_data, flush_sums);
        double total = 0.0;
        for (double value : flush_sums) { total += value; }
        flush_sink = total;
    };

    auto aligned_read_timed = [&] {
        aligned_read_kernel(thread_data, elements_per_thread);
    };
    auto unaligned_read_timed = [&] {
        unaligned_read_kernel(thread_data, elements_per_thread);
    };
    auto aligned_write_timed = [&] {
        aligned_write_kernel(thread_data, elements_per_thread, 1.0);
    };
    auto unaligned_write_timed = [&] {
        unaligned_write_kernel(thread_data, elements_per_thread, 1.0);
    };
    auto aligned_copy_timed = [&] {
        aligned_copy_kernel(thread_data, elements_per_thread);
    };
    auto unaligned_copy_timed = [&] {
        unaligned_copy_kernel(thread_data, elements_per_thread);
    };

    std::cout << "Measured bytes per iteration (reads/writes use full SIMD "
                 "blocks only): "
              << processed_bytes_all_threads
              << " B, copies: " << processed_copy_bytes_all_threads << " B"
              << std::endl;

    run_benchmark("Aligned read", processed_bytes_all_threads,
                  options.iterations, flush, aligned_read_timed, [] {});
    run_benchmark("Unaligned read", processed_bytes_all_threads,
                  options.iterations, flush, unaligned_read_timed, [] {});
    run_benchmark("Aligned write", processed_bytes_all_threads,
                  options.iterations, flush, aligned_write_timed, [] {});
    run_benchmark("Unaligned write", processed_bytes_all_threads,
                  options.iterations, flush, unaligned_write_timed, [] {});
    run_benchmark("Aligned copy", processed_copy_bytes_all_threads,
                  options.iterations, flush, aligned_copy_timed, [] {});
    run_benchmark("Unaligned copy", processed_copy_bytes_all_threads,
                  options.iterations, flush, unaligned_copy_timed, [] {});

    std::ofstream dev_null("/dev/null");
    dev_null << "Flush sink (ignore): " << flush_sink << std::endl;
    return 0;
}
