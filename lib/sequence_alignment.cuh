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

#include "affine_penalties.h"
#include "alignment_results.h"
#include "utils/sequences.h"

void allocate_offloaded_bt_d (wfa_backtrace_t** bt_offloaded_d,
                              const int max_steps,
                              const int num_blocks,
                              const size_t num_alignments);


size_t wf_data_buffer_size (const affine_penalties_t penalties,
                            const size_t max_steps,
                            const int band);

size_t wf_data_buffer_size_distance (const affine_penalties_t penalties,
                                     const size_t max_steps);

size_t available_shared_mem_per_block (const affine_penalties_t penalties,
                                       const size_t max_steps,
                                       const int threads_per_block);

void allocate_wf_data_buffer_d (uint8_t** wf_data_buffer,
                                const size_t max_steps,
                                const affine_penalties_t penalties,
                                const size_t num_blocks,
                                const int band);

void allocate_wf_data_buffer_distance_d (uint8_t** wf_data_buffer,
                                         const size_t max_steps,
                                         const affine_penalties_t penalties,
                                         const size_t num_blocks);


void reset_wf_data_buffer_distance_d (uint8_t* wf_data_buffer,
                                      const size_t max_steps,
                                      const affine_penalties_t penalties,
                                      const size_t num_blocks,
                                      cudaStream_t stream);

void launch_alignments_async (const char* packed_sequences_buffer,
                              const sequence_pair_t* sequences_metadata,
                              const size_t num_alignments,
                              const affine_penalties_t penalties,
                              alignment_result_t* const results,
                              wfa_backtrace_t* const backtraces,
                              alignment_result_t* const results_d,
                              wfa_backtrace_t* bt_offloaded_d,
                              uint8_t* const wf_data_buffer_d,
                              const int max_steps,
                              const int threads_per_block,
                              const int num_blocks,
                              size_t available_shared_mem_per_block,
                              int band,
                              cudaStream_t stream);

void copyInResults (alignment_result_t* const results,
                    const alignment_result_t* const results_d,
                    wfa_backtrace_t* const backtraces,
                    const wfa_backtrace_t* const bt_offloaded_d,
                    const size_t num_alignments,
                    const int max_steps,
                    const int num_blocks,
                    cudaStream_t stream);

void launch_alignments_distance_async (const char* packed_sequences_buffer,
                                       const sequence_pair_t* sequences_metadata,
                                       const size_t num_alignments,
                                       const affine_penalties_t penalties,
                                       alignment_result_t* const results,
                                       alignment_result_t *results_d,
                                       uint8_t* const wf_data_buffer,
                                       const int max_steps,
                                       const int threads_per_block,
                                       const int num_blocks,
                                       size_t available_shared_mem_per_block,
                                       int band,
                                       cudaStream_t stream);

void copyInResults_distance (alignment_result_t* const results,
                             const alignment_result_t* const results_d,
                             const size_t num_alignments,
                             cudaStream_t stream);
#endif
