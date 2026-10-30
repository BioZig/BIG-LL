// Apple Silicon Dedicated Metal GPU Batch Acceleration C-ABI Bridge
#ifndef BIGLL_METAL_BRIDGE_H
#define BIGLL_METAL_BRIDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct BigLLMetalContext BigLLMetalContext;

// Query whether Apple Silicon Metal compute is available on this system
bool bigll_metal_is_supported(void);

// Initialize Metal device, command queue, and precompile compute pipelines
BigLLMetalContext* bigll_metal_init(void);

// Destroy Metal context and release all GPU resources
void bigll_metal_deinit(BigLLMetalContext* ctx);

// Allocate a page-aligned shared zero-copy UMA buffer
void* bigll_metal_alloc_shared(BigLLMetalContext* ctx, size_t bytes);

// Free buffer allocated with bigll_metal_alloc_shared
void bigll_metal_free_shared(BigLLMetalContext* ctx, void* ptr, size_t bytes);

// Batch OneMax popcount evaluation on Apple Silicon GPU ALUs
// Supports arbitrary struct stride between chromosomes for zero-copy struct evaluation
bool bigll_metal_eval_onemax(
    BigLLMetalContext* ctx,
    const uint64_t* chrom_words,
    float* fitness_out,
    uint32_t num_words,
    uint32_t stride_words,
    uint32_t pop_size
);

// Batch Finance Markowitz mean-variance portfolio evaluation on Apple Silicon GPU ALUs
bool bigll_metal_eval_finance(
    BigLLMetalContext* ctx,
    const uint64_t* chrom_words,
    float* fitness_out,
    const float* expected_returns,
    const float* cov_matrix,
    uint32_t num_assets,
    float risk_aversion,
    uint32_t pop_size
);

#ifdef __cplusplus
}
#endif

#endif // BIGLL_METAL_BRIDGE_H
