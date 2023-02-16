//
// SPDX-License-Identifier: MIT
// Copyright (c) 2019 Andriy Berestovskyy <berestovskyy@gmail.com>
//
// Applied Benchmarks: Memory Latency
// Benchmarking Kaby Lake and Haswell memory latency using lists
//

#include <benchmark/benchmark.h>

#include <algorithm>
#include <iostream>
#include <random>

std::mt19937_64 m_generator{std::random_device()()};

// User-defined literals
auto constexpr operator"" _B(unsigned long long int n) { return n; }
auto constexpr operator"" _KB(unsigned long long int n) { return n * 1024; }
auto constexpr operator"" _M(unsigned long long int n) {
    return n * 1000 * 1000;
}

// Cache line size: 64 bytes for x86-64, 128 bytes for A64 ARMs
const auto kCachelineSize = 64_B;
// Memory page size. Default page size is 4 KB
const auto kPageSize = 4_KB;

// Singly linked list node with padding
struct Element {
    unsigned long index;
    std::byte     padding[kPageSize];
};

//
// Benchmark memory latency using a list.
//
// @param state.range(0)
//   Memory block size in KB to benchmark.
//
static void memory_latency_list(benchmark::State &state) {
    const auto mem_block_size = operator""_KB(state.range(0));
    // Each memory access fetches a cache line
    const auto num_nodes = mem_block_size / kCachelineSize;
    assert(num_nodes > 0);

    // Allocate a contiguous list of nodes for an iteration
    std::vector<Element> list(num_nodes);
    // initialise the array with sequential indices
    for (auto i = 0; i < num_nodes; ++i) { list[i].index = i; }
    // shuffle the array
    std::shuffle(list.begin(), list.end(), m_generator);
    const auto num_ops = 1_M;
    while (state.KeepRunningBatch(num_ops)) {
        auto index = 0UL;
        // iterate over num_ops
        for (auto i = 0; i < num_ops; ++i) {
            // access the next node
            index = list[index].index;
        }
        benchmark::DoNotOptimize(index);
    }

    state.counters["Size"] =
        benchmark::Counter(mem_block_size, benchmark::Counter::kDefaults,
                           benchmark::Counter::OneK::kIs1024);
    state.counters["Nodes"] =
        benchmark::Counter(num_nodes, benchmark::Counter::kDefaults,
                           benchmark::Counter::OneK::kIs1024);
    state.counters["Read Rate"] = benchmark::Counter(
        state.iterations() * kCachelineSize, benchmark::Counter::kIsRate,
        benchmark::Counter::OneK::kIs1024);
}

BENCHMARK(memory_latency_list)
    ->ArgName("size KB")
    ->RangeMultiplier(2)
    ->Range(1, 1 << 22);

BENCHMARK_MAIN();
