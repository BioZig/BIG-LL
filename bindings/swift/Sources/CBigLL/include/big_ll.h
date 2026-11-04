#ifndef BIG_LL_H
#define BIG_LL_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* big_ll_handle_t;

typedef double (*big_ll_eval_fn)(const uint64_t* genes, size_t word_count, void* user_data);

typedef struct {
    bool enable_dedup;
    size_t num_threads;
} big_ll_options_t;

big_ll_handle_t big_ll_create(
    size_t chromosome_bits,
    size_t pop_size,
    uint64_t seed,
    big_ll_eval_fn eval_fn,
    void* user_data
);

big_ll_handle_t big_ll_create_custom(
    size_t chromosome_bits,
    size_t pop_size,
    uint64_t seed,
    big_ll_eval_fn eval_fn,
    void* user_data,
    big_ll_options_t options
);

int big_ll_step(big_ll_handle_t handle);
int big_ll_run(big_ll_handle_t handle, size_t generations);

double big_ll_get_best_fitness(big_ll_handle_t handle);
size_t big_ll_get_best_genes(big_ll_handle_t handle, uint64_t* out_buffer, size_t buffer_words);
const uint64_t* big_ll_get_best_genes_ptr(big_ll_handle_t handle);

size_t big_ll_get_generation(big_ll_handle_t handle);
size_t big_ll_get_pop_size(big_ll_handle_t handle);
size_t big_ll_get_chromosome_bits(big_ll_handle_t handle);
size_t big_ll_get_word_count(big_ll_handle_t handle);

const uint64_t* big_ll_get_chromosome_words(big_ll_handle_t handle, size_t index);
double big_ll_get_chromosome_fitness(big_ll_handle_t handle, size_t index);

void big_ll_destroy(big_ll_handle_t handle);

#ifdef __cplusplus
}
#endif

#endif /* BIG_LL_H */
