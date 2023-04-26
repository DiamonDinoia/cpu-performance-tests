#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

// optional wrapper if you don't want to just use __rdtsc() everywhere
inline __attribute__((always_inline)) unsigned long long readTSC() {
    _mm_lfence();  // optionally wait for earlier insns to retire before reading
                   // the clock
    auto result = __rdtsc();
    _mm_lfence();  // optionally block later instructions until rdtsc retires
    return result;
}

std::mt19937_64                 m_generator{std::random_device()()};
std::uniform_int_distribution<> distrib{0, 100};

// User-defined literals
auto constexpr operator"" _B(unsigned long long int n) { return n; }
auto constexpr operator"" _KB(unsigned long long int n) { return n * 1024; }
auto constexpr operator"" _M(unsigned long long int n) {
    return n * 1000 * 1000;
}

// This function check is the csv file exists
// if it does not exist it creates it and writes the header
// the header contains the name branching probability and the number of
// instructions per cycle if the file exists it appends the branching
// probability and the number of instructions per cycle
static void write_csv(const unsigned branch_prob,
                      const double   instructions_per_cycle,
                      const bool    branchless = false
                      ) {
    std::ofstream file;
    file.open("branch.csv", std::ios_base::app);
    if (file.tellp() == 0) { file << "probability,cycles,branchless" << std::endl; }
    file << branch_prob << ", " << instructions_per_cycle <<"," << branchless << std::endl;
    file.close();
}

// This function is used to benchmark the branching
// it takes a branch_prob parameter which is the probability of the branch being
// taken it measures the number of instructions per cycle
static void branching_benchmark(const unsigned branch_prob) {
    const auto                 num_ops = 100_M;
    std::vector<unsigned long> list(num_ops);
    // initialise the array with the distribution
    for (auto i = 0; i < num_ops; ++i) { list[i] = distrib(m_generator); }

    volatile int sum   = 0;
    const auto   start = readTSC();
    // sum the array into sum
    for (auto i = 0; i < num_ops; ++i) {
        if (list[i] < branch_prob) { sum += list[i]; }
    }
    const auto stop = readTSC();
    std::cout << "branching_benchmark(" << branch_prob << ") took "
              << double(stop - start) / num_ops << " cycles per iteration"
              << std::endl;
    write_csv(branch_prob, double(stop - start) / num_ops);
}

// This function is used to benchmark the branchless version
// it takes a branch_prob parameter which is the probability of the branch being
// taken it measures the number of instructions per cycle
static void branchless_benchmark(const unsigned branch_prob) {
    const auto                 num_ops = 100_M;
    std::vector<unsigned long> list(num_ops);
    // initialise the array with the distribution
    for (auto i = 0; i < num_ops; ++i) { list[i] = distrib(m_generator); }

    volatile int sum   = 0;
    const auto   start = readTSC();
    // sum the array into sum
    for (auto i = 0; i < num_ops; ++i) {
        sum += list[i] * (list[i] < branch_prob);
    }
    const auto stop = readTSC();
    std::cout << "branchless_benchmark(" << branch_prob << ") took "
              << double(stop - start) / num_ops << " cycles per iteration"
              << std::endl;
    write_csv(branch_prob, double(stop - start) / num_ops, true);
}

int main() {
    for (size_t i = 0; i < 101; i++) { branching_benchmark(i); }
    for (size_t i = 0; i < 101; i++) { branchless_benchmark(i); }
    return 0;
}