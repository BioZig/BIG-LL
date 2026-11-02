// Apple Silicon Dedicated Metal GPU Batch Acceleration Objective-C / Metal Bridge
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "metal_bridge.h"
#include <stdlib.h>
#include <string.h>

static const char* kBigLLMetalShaderSource =
"#include <metal_stdlib>\n"
"using namespace metal;\n"
"\n"
"kernel void onemax_eval_kernel(\n"
"    device const ulong* chromosomes [[buffer(0)]],\n"
"    device float* fitnesses [[buffer(1)]],\n"
"    constant uint& num_words [[buffer(2)]],\n"
"    constant uint& stride_words [[buffer(3)]],\n"
"    constant uint& pop_size [[buffer(4)]],\n"
"    uint tid [[thread_position_in_grid]])\n"
"{\n"
"    if (tid >= pop_size) return;\n"
"    uint sum = 0;\n"
"    device const ulong* chrom = chromosomes + tid * stride_words;\n"
"    #pragma unroll\n"
"    for (uint i = 0; i < num_words; ++i) {\n"
"        sum += popcount(chrom[i]);\n"
"    }\n"
"    fitnesses[tid] = (float)sum;\n"
"}\n"
"\n"
"kernel void finance_eval_kernel(\n"
"    device const ulong* chromosomes [[buffer(0)]],\n"
"    device float* fitnesses [[buffer(1)]],\n"
"    constant float* expected_returns [[buffer(2)]],\n"
"    constant float* cov_matrix [[buffer(3)]],\n"
"    constant uint& num_assets [[buffer(4)]],\n"
"    constant float& risk_aversion [[buffer(5)]],\n"
"    constant uint& pop_size [[buffer(6)]],\n"
"    uint tid [[thread_position_in_grid]])\n"
"{\n"
"    if (tid >= pop_size) return;\n"
"    ulong chrom = chromosomes[tid];\n"
"    float4 w_vec[4];\n"
"    float sum_w = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint k = 0; k < 4; ++k) {\n"
"        uint chunk = (uint)((chrom >> (k * 16)) & 0xFFFF);\n"
"        float4 wv = float4(\n"
"            (float)(chunk & 0xF),\n"
"            (float)((chunk >> 4) & 0xF),\n"
"            (float)((chunk >> 8) & 0xF),\n"
"            (float)((chunk >> 12) & 0xF)\n"
"        );\n"
"        w_vec[k] = wv;\n"
"        sum_w += wv.x + wv.y + wv.z + wv.w;\n"
"    }\n"
"    if (sum_w <= 0.0001f) {\n"
"        fitnesses[tid] = -1000.0f;\n"
"        return;\n"
"    }\n"
"    float inv_sum = 1.0f / sum_w;\n"
"    #pragma unroll\n"
"    for (uint k = 0; k < 4; ++k) {\n"
"        w_vec[k] *= inv_sum;\n"
"    }\n"
"    constant const float4* ret_vec = (constant const float4*)expected_returns;\n"
"    float port_ret = dot(w_vec[0], ret_vec[0]) + dot(w_vec[1], ret_vec[1])\n"
"                   + dot(w_vec[2], ret_vec[2]) + dot(w_vec[3], ret_vec[3]);\n"
"    constant const float4* cov_vec = (constant const float4*)cov_matrix;\n"
"    float port_var = 0.0f;\n"
"    #pragma unroll\n"
"    for (uint i = 0; i < 16; ++i) {\n"
"        float row_dot = dot(cov_vec[i * 4 + 0], w_vec[0])\n"
"                      + dot(cov_vec[i * 4 + 1], w_vec[1])\n"
"                      + dot(cov_vec[i * 4 + 2], w_vec[2])\n"
"                      + dot(cov_vec[i * 4 + 3], w_vec[3]);\n"
"        float wi = (i < 4) ? w_vec[0][i] : ((i < 8) ? w_vec[1][i - 4] : ((i < 12) ? w_vec[2][i - 8] : w_vec[3][i - 12]));\n"
"        port_var = fma(wi, row_dot, port_var);\n"
"    }\n"
"    fitnesses[tid] = port_ret - (0.5f * risk_aversion * port_var);\n"
"}\n"
"\n"
"inline uint lcg_next(thread uint& state) {\n"
"    state = state * 1664525u + 1013904223u;\n"
"    return state;\n"
"}\n"
"\n"
"kernel void ga_reproduce_kernel(\n"
"    device const ulong* current_pop [[buffer(0)]],\n"
"    device const float* fitnesses [[buffer(1)]],\n"
"    device ulong* next_pop [[buffer(2)]],\n"
"    constant uint& pop_size [[buffer(3)]],\n"
"    constant uint& tournament_size [[buffer(4)]],\n"
"    constant float& mutation_rate [[buffer(5)]],\n"
"    constant uint& seed [[buffer(6)]],\n"
"    uint tid [[thread_position_in_grid]])\n"
"{\n"
"    if (tid >= pop_size) return;\n"
"    uint rng = seed + tid * 2654435761u;\n"
"    rng = lcg_next(rng);\n"
"    uint p1_idx = lcg_next(rng) % pop_size;\n"
"    float best_f1 = fitnesses[p1_idx];\n"
"    #pragma unroll\n"
"    for (uint k = 1; k < 4; ++k) {\n"
"        if (k >= tournament_size) break;\n"
"        uint cand = lcg_next(rng) % pop_size;\n"
"        float f = fitnesses[cand];\n"
"        if (f > best_f1) {\n"
"            best_f1 = f;\n"
"            p1_idx = cand;\n"
"        }\n"
"    }\n"
"    uint p2_idx = lcg_next(rng) % pop_size;\n"
"    float best_f2 = fitnesses[p2_idx];\n"
"    #pragma unroll\n"
"    for (uint k = 1; k < 4; ++k) {\n"
"        if (k >= tournament_size) break;\n"
"        uint cand = lcg_next(rng) % pop_size;\n"
"        float f = fitnesses[cand];\n"
"        if (f > best_f2) {\n"
"            best_f2 = f;\n"
"            p2_idx = cand;\n"
"        }\n"
"    }\n"
"    ulong p1 = current_pop[p1_idx];\n"
"    ulong p2 = current_pop[p2_idx];\n"
"    ulong mask = ((ulong)lcg_next(rng) << 32) | (ulong)lcg_next(rng);\n"
"    ulong child = (p1 & mask) | (p2 & ~mask);\n"
"    uint mut_rand = lcg_next(rng);\n"
"    if ((float)(mut_rand & 0xFFFF) / 65536.0f < mutation_rate) {\n"
"        uint bit_to_flip = lcg_next(rng) % 64;\n"
"        child ^= (1ULL << bit_to_flip);\n"
"    }\n"
"    next_pop[tid] = child;\n"
"}\n";

struct BigLLMetalContext {
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLComputePipelineState> onemax_pipeline;
    id<MTLComputePipelineState> finance_pipeline;
    id<MTLComputePipelineState> reproduce_pipeline;

    // Persistent zero-copy UMA shared buffers
    id<MTLBuffer> shared_chrom_buf;
    id<MTLBuffer> shared_chrom_buf_b;
    id<MTLBuffer> shared_fit_buf;
    size_t shared_chrom_capacity_bytes;
    size_t shared_fit_capacity_bytes;

    // Cached constant parameter buffers
    id<MTLBuffer> finance_ret_buf;
    id<MTLBuffer> finance_cov_buf;
};

bool bigll_metal_is_supported(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return false;
        return [dev hasUnifiedMemory];
    }
}

BigLLMetalContext* bigll_metal_init(void) {
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) return NULL;

        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:kBigLLMetalShaderSource];
        id<MTLLibrary> library = [dev newLibraryWithSource:src options:nil error:&error];
        if (!library) {
            NSLog(@"[BIG-LL Metal] Failed to compile MSL shader: %@", [error localizedDescription]);
            return NULL;
        }

        id<MTLFunction> onemax_fn = [library newFunctionWithName:@"onemax_eval_kernel"];
        id<MTLComputePipelineState> onemax_pso = [dev newComputePipelineStateWithFunction:onemax_fn error:&error];
        if (!onemax_pso) {
            NSLog(@"[BIG-LL Metal] Failed to create OneMax pipeline: %@", [error localizedDescription]);
            return NULL;
        }

        id<MTLFunction> finance_fn = [library newFunctionWithName:@"finance_eval_kernel"];
        id<MTLComputePipelineState> finance_pso = [dev newComputePipelineStateWithFunction:finance_fn error:&error];
        if (!finance_pso) {
            NSLog(@"[BIG-LL Metal] Failed to create Finance pipeline: %@", [error localizedDescription]);
            return NULL;
        }

        id<MTLFunction> repro_fn = [library newFunctionWithName:@"ga_reproduce_kernel"];
        id<MTLComputePipelineState> repro_pso = [dev newComputePipelineStateWithFunction:repro_fn error:&error];
        if (!repro_pso) {
            NSLog(@"[BIG-LL Metal] Failed to create Reproduce pipeline: %@", [error localizedDescription]);
            return NULL;
        }

        id<MTLCommandQueue> queue = [dev newCommandQueue];
        if (!queue) return NULL;

        BigLLMetalContext* ctx = (BigLLMetalContext*)calloc(1, sizeof(BigLLMetalContext));
        ctx->device = dev;
        ctx->command_queue = queue;
        ctx->onemax_pipeline = onemax_pso;
        ctx->finance_pipeline = finance_pso;
        ctx->reproduce_pipeline = repro_pso;

        // Preallocate 8MB shared buffers
        ctx->shared_chrom_capacity_bytes = 131072 * 8 * sizeof(uint64_t); // 8 MB
        ctx->shared_fit_capacity_bytes = 131072 * sizeof(float);           // 512 KB

        ctx->shared_chrom_buf = [dev newBufferWithLength:ctx->shared_chrom_capacity_bytes
                                                 options:MTLResourceStorageModeShared];
        ctx->shared_chrom_buf_b = [dev newBufferWithLength:ctx->shared_chrom_capacity_bytes
                                                   options:MTLResourceStorageModeShared];
        ctx->shared_fit_buf = [dev newBufferWithLength:ctx->shared_fit_capacity_bytes
                                               options:MTLResourceStorageModeShared];

        ctx->finance_ret_buf = [dev newBufferWithLength:16 * sizeof(float)
                                                options:MTLResourceStorageModeShared];
        ctx->finance_cov_buf = [dev newBufferWithLength:256 * sizeof(float)
                                                options:MTLResourceStorageModeShared];

        return ctx;
    }
}

void bigll_metal_deinit(BigLLMetalContext* ctx) {
    if (!ctx) return;
    @autoreleasepool {
        ctx->shared_chrom_buf = nil;
        ctx->shared_chrom_buf_b = nil;
        ctx->shared_fit_buf = nil;
        ctx->finance_ret_buf = nil;
        ctx->finance_cov_buf = nil;
        ctx->onemax_pipeline = nil;
        ctx->finance_pipeline = nil;
        ctx->reproduce_pipeline = nil;
        ctx->command_queue = nil;
        ctx->device = nil;
        free(ctx);
    }
}

void* bigll_metal_alloc_shared(BigLLMetalContext* ctx, size_t bytes) {
    if (!ctx || bytes == 0) return NULL;
    @autoreleasepool {
        id<MTLBuffer> buf = [ctx->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (!buf) return NULL;
        return [buf contents];
    }
}

void bigll_metal_free_shared(BigLLMetalContext* ctx, void* ptr, size_t bytes) {
    (void)ctx;
    (void)ptr;
    (void)bytes;
}

static void ensure_chrom_buffer(BigLLMetalContext* ctx, size_t required_bytes) {
    if (required_bytes > ctx->shared_chrom_capacity_bytes) {
        ctx->shared_chrom_capacity_bytes = required_bytes * 2;
        ctx->shared_chrom_buf = [ctx->device newBufferWithLength:ctx->shared_chrom_capacity_bytes
                                                         options:MTLResourceStorageModeShared];
        ctx->shared_chrom_buf_b = [ctx->device newBufferWithLength:ctx->shared_chrom_capacity_bytes
                                                           options:MTLResourceStorageModeShared];
    }
}

static void ensure_fit_buffer(BigLLMetalContext* ctx, size_t required_bytes) {
    if (required_bytes > ctx->shared_fit_capacity_bytes) {
        ctx->shared_fit_capacity_bytes = required_bytes * 2;
        ctx->shared_fit_buf = [ctx->device newBufferWithLength:ctx->shared_fit_capacity_bytes
                                                       options:MTLResourceStorageModeShared];
    }
}

void* bigll_metal_get_shared_chrom_buffer(BigLLMetalContext* ctx, size_t required_bytes) {
    if (!ctx) return NULL;
    @autoreleasepool {
        ensure_chrom_buffer(ctx, required_bytes);
        return [ctx->shared_chrom_buf contents];
    }
}

void* bigll_metal_get_shared_fit_buffer(BigLLMetalContext* ctx, size_t required_bytes) {
    if (!ctx) return NULL;
    @autoreleasepool {
        ensure_fit_buffer(ctx, required_bytes);
        return [ctx->shared_fit_buf contents];
    }
}

bool bigll_metal_eval_onemax(
    BigLLMetalContext* ctx,
    const uint64_t* chrom_words,
    float* fitness_out,
    uint32_t num_words,
    uint32_t stride_words,
    uint32_t pop_size
) {
    if (!ctx || !chrom_words || !fitness_out || pop_size == 0 || num_words == 0 || stride_words == 0) return false;

    @autoreleasepool {
        size_t chrom_bytes = (size_t)pop_size * stride_words * sizeof(uint64_t);
        size_t fit_bytes = (size_t)pop_size * sizeof(float);

        ensure_chrom_buffer(ctx, chrom_bytes);
        ensure_fit_buffer(ctx, fit_bytes);

        if (chrom_words != [ctx->shared_chrom_buf contents]) {
            memcpy([ctx->shared_chrom_buf contents], chrom_words, chrom_bytes);
        }

        id<MTLCommandBuffer> cmd_buf = [ctx->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:ctx->onemax_pipeline];
        [encoder setBuffer:ctx->shared_chrom_buf offset:0 atIndex:0];
        [encoder setBuffer:ctx->shared_fit_buf offset:0 atIndex:1];
        [encoder setBytes:&num_words length:sizeof(uint32_t) atIndex:2];
        [encoder setBytes:&stride_words length:sizeof(uint32_t) atIndex:3];
        [encoder setBytes:&pop_size length:sizeof(uint32_t) atIndex:4];

        MTLSize gridSize = MTLSizeMake(pop_size, 1, 1);
        NSUInteger threadGroupSize = ctx->onemax_pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > 256) threadGroupSize = 256;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        if (fitness_out != [ctx->shared_fit_buf contents]) {
            memcpy(fitness_out, [ctx->shared_fit_buf contents], fit_bytes);
        }

        return true;
    }
}

bool bigll_metal_eval_finance(
    BigLLMetalContext* ctx,
    const uint64_t* chrom_words,
    float* fitness_out,
    const float* expected_returns,
    const float* cov_matrix,
    uint32_t num_assets,
    float risk_aversion,
    uint32_t pop_size
) {
    if (!ctx || !chrom_words || !fitness_out || pop_size == 0) return false;
    if (num_assets > 16) return false;

    @autoreleasepool {
        size_t chrom_bytes = (size_t)pop_size * sizeof(uint64_t);
        size_t fit_bytes = (size_t)pop_size * sizeof(float);

        ensure_chrom_buffer(ctx, chrom_bytes);
        ensure_fit_buffer(ctx, fit_bytes);

        if (chrom_words != [ctx->shared_chrom_buf contents]) {
            memcpy([ctx->shared_chrom_buf contents], chrom_words, chrom_bytes);
        }

        memcpy([ctx->finance_ret_buf contents], expected_returns, num_assets * sizeof(float));
        memcpy([ctx->finance_cov_buf contents], cov_matrix, num_assets * num_assets * sizeof(float));

        id<MTLCommandBuffer> cmd_buf = [ctx->command_queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];

        [encoder setComputePipelineState:ctx->finance_pipeline];
        [encoder setBuffer:ctx->shared_chrom_buf offset:0 atIndex:0];
        [encoder setBuffer:ctx->shared_fit_buf offset:0 atIndex:1];
        [encoder setBuffer:ctx->finance_ret_buf offset:0 atIndex:2];
        [encoder setBuffer:ctx->finance_cov_buf offset:0 atIndex:3];
        [encoder setBytes:&num_assets length:sizeof(uint32_t) atIndex:4];
        [encoder setBytes:&risk_aversion length:sizeof(float) atIndex:5];
        [encoder setBytes:&pop_size length:sizeof(uint32_t) atIndex:6];

        MTLSize gridSize = MTLSizeMake(pop_size, 1, 1);
        NSUInteger threadGroupSize = ctx->finance_pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > 256) threadGroupSize = 256;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
        [encoder endEncoding];

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        if (fitness_out != [ctx->shared_fit_buf contents]) {
            memcpy(fitness_out, [ctx->shared_fit_buf contents], fit_bytes);
        }

        return true;
    }
}

bool bigll_metal_evolve_finance(
    BigLLMetalContext* ctx,
    uint32_t pop_size,
    uint32_t num_generations,
    uint32_t tournament_size,
    float mutation_rate,
    const float* expected_returns,
    const float* cov_matrix,
    uint32_t num_assets,
    float risk_aversion,
    uint32_t seed,
    float* best_fitness_out
) {
    if (!ctx || pop_size == 0 || num_generations == 0) return false;
    if (num_assets > 16) return false;

    @autoreleasepool {
        size_t chrom_bytes = (size_t)pop_size * sizeof(uint64_t);
        size_t fit_bytes = (size_t)pop_size * sizeof(float);

        ensure_chrom_buffer(ctx, chrom_bytes);
        ensure_fit_buffer(ctx, fit_bytes);

        memcpy([ctx->finance_ret_buf contents], expected_returns, num_assets * sizeof(float));
        memcpy([ctx->finance_cov_buf contents], cov_matrix, num_assets * num_assets * sizeof(float));

        id<MTLBuffer> cur_pop = ctx->shared_chrom_buf;
        id<MTLBuffer> next_pop = ctx->shared_chrom_buf_b;

        MTLSize gridSize = MTLSizeMake(pop_size, 1, 1);
        NSUInteger threadGroupSize = ctx->finance_pipeline.maxTotalThreadsPerThreadgroup;
        if (threadGroupSize > 256) threadGroupSize = 256;
        MTLSize threadgroupSize = MTLSizeMake(threadGroupSize, 1, 1);

        id<MTLCommandBuffer> cmd_buf = [ctx->command_queue commandBuffer];

        // Generation 0 initial evaluation
        {
            id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
            [encoder setComputePipelineState:ctx->finance_pipeline];
            [encoder setBuffer:cur_pop offset:0 atIndex:0];
            [encoder setBuffer:ctx->shared_fit_buf offset:0 atIndex:1];
            [encoder setBuffer:ctx->finance_ret_buf offset:0 atIndex:2];
            [encoder setBuffer:ctx->finance_cov_buf offset:0 atIndex:3];
            [encoder setBytes:&num_assets length:sizeof(uint32_t) atIndex:4];
            [encoder setBytes:&risk_aversion length:sizeof(float) atIndex:5];
            [encoder setBytes:&pop_size length:sizeof(uint32_t) atIndex:6];
            [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
            [encoder endEncoding];
        }

        // Generational evolutionary loop entirely on GPU ALUs
        for (uint32_t g = 1; g < num_generations; ++g) {
            uint32_t gen_seed = seed + g * 1013904223u;

            // 1. Selection + Crossover + Mutation: cur_pop -> next_pop
            {
                id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
                [encoder setComputePipelineState:ctx->reproduce_pipeline];
                [encoder setBuffer:cur_pop offset:0 atIndex:0];
                [encoder setBuffer:ctx->shared_fit_buf offset:0 atIndex:1];
                [encoder setBuffer:next_pop offset:0 atIndex:2];
                [encoder setBytes:&pop_size length:sizeof(uint32_t) atIndex:3];
                [encoder setBytes:&tournament_size length:sizeof(uint32_t) atIndex:4];
                [encoder setBytes:&mutation_rate length:sizeof(float) atIndex:5];
                [encoder setBytes:&gen_seed length:sizeof(uint32_t) atIndex:6];
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // 2. Evaluation: next_pop -> shared_fit_buf
            {
                id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
                [encoder setComputePipelineState:ctx->finance_pipeline];
                [encoder setBuffer:next_pop offset:0 atIndex:0];
                [encoder setBuffer:ctx->shared_fit_buf offset:0 atIndex:1];
                [encoder setBuffer:ctx->finance_ret_buf offset:0 atIndex:2];
                [encoder setBuffer:ctx->finance_cov_buf offset:0 atIndex:3];
                [encoder setBytes:&num_assets length:sizeof(uint32_t) atIndex:4];
                [encoder setBytes:&risk_aversion length:sizeof(float) atIndex:5];
                [encoder setBytes:&pop_size length:sizeof(uint32_t) atIndex:6];
                [encoder dispatchThreads:gridSize threadsPerThreadgroup:threadgroupSize];
                [encoder endEncoding];
            }

            // Ping-pong buffer swap
            id<MTLBuffer> tmp = cur_pop;
            cur_pop = next_pop;
            next_pop = tmp;
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];

        if (best_fitness_out) {
            const float* fits = (const float*)[ctx->shared_fit_buf contents];
            float max_f = fits[0];
            for (size_t i = 1; i < pop_size; ++i) {
                if (fits[i] > max_f) max_f = fits[i];
            }
            *best_fitness_out = max_f;
        }

        return true;
    }
}
