/*
 * Copyright (c) 2021 Quim Aguado
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#ifndef SEQUENCE_ALIGNMENT_CUH
#define SEQUENCE_ALIGNMENT_CUH

#include <stdint.h>
#include "wfa_types.h"
#include "affine_penalties.h"
#include "alignment_results.h"
#include "utils/sequences.h"

#define EWAVEFRONT_V(k,offset) ((offset)-(k))
#define EWAVEFRONT_H(k,offset) (offset)
#define EWAVEFRONT_DIAGONAL(h,v) ((h)-(v))
#define EWAVEFRONT_OFFSET(h,v)   (h)

typedef struct {
    int16_t hi;
    int16_t lo;
    // At position 0, pointer to offsets in shared memory, at position 1,
    // offsets at global memory
    wfa_offset_t* offsets[2];
    bool exist;
    // {backtrace vector, prev pointer} per diagonal
    wfa_backtrace_t* bt;
} wfa_wavefront_t;

// Elements of one global working-set array (offsets or backtraces) holding
// `num_wfs` wavefronts. Padded so every array starts 32-bit aligned.
__host__ __device__ inline int alignment_wf_elements (const int num_wfs,
                                                      const int max_steps) {
    const int n = num_wfs * (2 * max_steps + 1);
    return n + (4 - (n % 4));
}

// Bytes of global working set used by each alignment_kernel block: ~M keeps
// max(o+e, x)+1 wavefronts, ~I and ~D only e+1 (they are read at s-e).
// Shared by the host allocation and the kernel's per-block layout.
__host__ __device__ inline size_t alignment_wf_data_buffer_size (
                                            const affine_penalties_t penalties,
                                            const int max_steps) {
    const int aws = max(penalties.o + penalties.e, penalties.x) + 1;
    const int ws_id = penalties.e + 1;
    const size_t elements = alignment_wf_elements(aws, max_steps)
                            + 2 * (size_t)alignment_wf_elements(ws_id, max_steps);
    return elements * (sizeof(wfa_offset_t) + sizeof(wfa_backtrace_t));
}

__global__ void alignment_kernel (
                            const char* packed_sequences_buffer,
                            const sequence_pair_t* sequences_metadata,
                            const size_t num_alignments,
                            const int max_steps,
                            uint8_t* const wf_data_buffer,
                            const affine_penalties_t penalties,
                            wfa_backtrace_t* offloaded_backtraces_global,
                            wfa_backtrace_t* offloaded_backtraces_results,
                            alignment_result_t* results,
                            uint32_t* const next_alignment_idx,
                            const size_t num_sh_offsets_per_wf);

#endif
