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

#include "kernels/sequence_alignment_kernel.cuh"
#include "kernels/sequence_alignment_kernel_aband.cuh"
#include "kernels/sequence_distance_kernel.cuh"
#include "kernels/sequence_distance_kernel_aband.cuh"
#include "wfa_types.h"
#include "utils/cuda_utils.cuh"
#include "utils/logger.h"
#include "sequence_alignment.cuh"

void allocate_offloaded_bt_d (wfa_backtrace_t** bt_offloaded_d,
                              const int max_steps,
                              const int num_blocks,
                              const size_t num_alignments) {
    // Backtrace indices (bt_prev_t) are relative to each block's own region
    // of the offloaded backtraces array, so the addressability check applies
    // to the per-block size, not to the total.
    size_t bt_offloaded_size = BT_OFFLOADED_ELEMENTS(max_steps);
    size_t max_addressable_elements = (size_t)1 << (sizeof(bt_prev_t) * 8);

    if (bt_offloaded_size >= max_addressable_elements) {
        LOG_ERROR("Trying to allocate more backtrace elements than the ones"
                  " that we can address.")
        exit(-1);
    }

    // The total across all blocks can exceed INT_MAX: keep it in size_t.
    bt_offloaded_size *= (size_t)num_blocks;

    // Add the results array
    size_t bt_offloaded_results_size = BT_OFFLOADED_RESULT_ELEMENTS(max_steps)
                                       * num_alignments;

    LOG_DEBUG("Allocating %.2f MiB to store backtraces of %zu alignments using %d blocks.",
              (float)((bt_offloaded_size + bt_offloaded_results_size) * sizeof(wfa_backtrace_t)) / (1 << 20),
              num_alignments, num_blocks)

    cudaMalloc(bt_offloaded_d,
               (bt_offloaded_size + bt_offloaded_results_size) * sizeof(wfa_backtrace_t));
    CUDA_CHECK_ERR
}


size_t available_shared_mem_per_block (const affine_penalties_t penalties,
                                       const size_t max_steps,
                                       const int threads_per_block) {
    // TODO: Take band into account
    // TODO: Choose device from argument
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    CUDA_CHECK_ERR

    const size_t shared_mem_per_block = deviceProp.sharedMemPerBlock;
    const size_t shared_mem_per_multiprocessor = deviceProp.sharedMemPerMultiprocessor;
    const int max_occupancy = deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;

    // Assume we get an occupancy of 32 warps per SM, with the limiting factor
    // being the registers used.
    const int max_active_warps_per_sm = min(32, max_occupancy);

    const int warps_per_block = max(1, threads_per_block / deviceProp.warpSize);
    // Small blocks are limited by the maximum resident blocks per SM
    const int blocks_per_sm = max(1, min(max_active_warps_per_sm / warps_per_block,
                                         deviceProp.maxBlocksPerMultiProcessor));
    const size_t usable_sh_mem_per_block = min(shared_mem_per_multiprocessor / blocks_per_sm,
                                               shared_mem_per_block);

    LOG_DEBUG("Maximum usable shared memory per multiprocessor: %.2fKiB. Maximum usable shared memory per block: %.2fKiB. Maximum shared memory per block=%.2fKiB",
              (double)shared_mem_per_multiprocessor/ (1<<10),
              (double)usable_sh_mem_per_block / (1<<10),
              (double)shared_mem_per_block / (1<<10))
    return usable_sh_mem_per_block;
}

size_t wf_data_buffer_size (const affine_penalties_t penalties,
                             const size_t max_steps,
                             const int band) {
    if (band <= 0) return alignment_wf_data_buffer_size(penalties, max_steps);

    // Banded kernel: 3 rings of max(o+e, x)+1 wavefronts each
    const int max_wf_size = 2 * max_steps + 1;
    const int active_working_set = max(penalties.o+penalties.e, penalties.x) + 1;
    int offsets_elements = active_working_set * max_wf_size;
    offsets_elements = offsets_elements + (4 - (offsets_elements % 4));
    const int bt_elements = offsets_elements;
    size_t buffer_size =
                    // Offsets space
                    (offsets_elements * 3 * sizeof(wfa_offset_t))
                    // Backtraces vectors
                    + (bt_elements * 3 * sizeof(bt_vector_t))
                    // Backtraces pointers
                    + (bt_elements * 3 * sizeof(bt_prev_t));
    return buffer_size;
}

size_t wf_data_buffer_size_distance (const affine_penalties_t penalties,
                                     const size_t max_steps) {
    const int max_wf_size = 2 * max_steps + 1;
    const int active_working_set = max(penalties.o+penalties.e, penalties.x) + 1;
    int offsets_elements = active_working_set * max_wf_size;
    offsets_elements = offsets_elements + (4 - (offsets_elements % 4));
    size_t buffer_size = (offsets_elements * 3 * sizeof(wfa_offset_t));
    return buffer_size;
}

void allocate_wf_data_buffer_d (uint8_t** wf_data_buffer,
                                const size_t max_steps,
                                const affine_penalties_t penalties,
                                const size_t num_blocks,
                                const int band) {

    // Create the active working set buffer on global memory
    size_t buffer_size = wf_data_buffer_size(penalties, max_steps, band);
    LOG_DEBUG("Working set size per block: %.2f KiB",
              (float)(buffer_size) / (1 << 10))
    buffer_size *= num_blocks;

    // Add a single int to be the global index of the next alignment in the pool
    buffer_size += sizeof(uint32_t);

    LOG_DEBUG("Allocating %.2f MiB to store working set data of %zu workers.",
              (float)(buffer_size) / (1 << 20), num_blocks)

    cudaMalloc(wf_data_buffer, buffer_size);
    CUDA_CHECK_ERR;
}

void allocate_wf_data_buffer_distance_d (uint8_t** wf_data_buffer,
                                         const size_t max_steps,
                                         const affine_penalties_t penalties,
                                         const size_t num_blocks) {

    // Create the active working set buffer on global memory
    size_t buffer_size = wf_data_buffer_size_distance(penalties, max_steps);
    LOG_DEBUG("Working set size per block: %.2f KiB",
              (float)(buffer_size) / (1 << 10))
    buffer_size *= num_blocks;

    // Add a single int to be the global index of the next alignment in the pool
    buffer_size += sizeof(uint32_t);

    LOG_DEBUG("Allocating %.2f MiB to store working set data of %zu workers.",
              (float)(buffer_size) / (1 << 20), num_blocks)

    cudaMalloc(wf_data_buffer, buffer_size);
    CUDA_CHECK_ERR;
}


void reset_wf_data_buffer_distance_d (uint8_t* wf_data_buffer,
                                      const size_t max_steps,
                                      const affine_penalties_t penalties,
                                      const size_t num_blocks,
                                      cudaStream_t stream) {

    if (wf_data_buffer == NULL) return;
    // Create the active working set buffer on global memory
    size_t buffer_size = wf_data_buffer_size_distance(penalties, max_steps);
    buffer_size *= num_blocks;
    buffer_size += sizeof(uint32_t);

    cudaMemsetAsync(wf_data_buffer, 0, buffer_size, stream);
    CUDA_CHECK_ERR;
}

void launch_alignments_async (const char* packed_sequences_buffer,
                              const sequence_pair_t* sequences_metadata,
                              const size_t num_alignments,
                              const affine_penalties_t penalties,
                              alignment_result_t* const results,
                              wfa_backtrace_t* const backtraces,
                              alignment_result_t *results_d,
                              wfa_backtrace_t* bt_offloaded_d,
                              uint8_t* const wf_data_buffer,
                              const int max_steps,
                              const int threads_per_block,
                              const int num_blocks,
                              size_t available_sh_mem_per_block,
                              int band,
                              cudaStream_t stream) {
    size_t bt_offloaded_size = BT_OFFLOADED_ELEMENTS(max_steps) * (size_t)num_blocks;

    wfa_backtrace_t* bt_offloaded_results_d = bt_offloaded_d
                                              + bt_offloaded_size;

    // TODO: Reduction of penalties
    const int max_wf_size = 2 * max_steps + 1;
    const int active_working_set = max(penalties.o+penalties.e, penalties.x) + 1;
    int offsets_elements = active_working_set * max_wf_size;
    offsets_elements = offsets_elements + (4 - (offsets_elements % 4));

    size_t sh_mem_size;
    if (band > 0) {
        sh_mem_size = \
                    // Wavefronts structs space
                    + (active_working_set * sizeof(wfa_aband_wavefront_t) * 3)
                    // Position of the last used element in the offloaded
                    // backtraces. It will be atomically increased.
                    + sizeof(int);
    } else {
        sh_mem_size = \
                    // Wavefronts structs space (~I/~D rings have e+1 slots)
                    + ((active_working_set + 2 * (penalties.e + 1)) * sizeof(wfa_wavefront_t))
                    // Position of the last used element in the offloaded
                    // backtraces. It will be atomically increased.
                    + sizeof(int);

    }

    // The wavefront structs alone may not fit in the per-block budget (small
    // blocks, large penalties): then keep only the minimum shared offsets
    available_sh_mem_per_block = (available_sh_mem_per_block > sh_mem_size)
                                 ? (available_sh_mem_per_block - sh_mem_size) : 0;
    // Using 100% of the shared memory available can give some problems, as the
    // driver sometimes take some sh memory space (?)
    available_sh_mem_per_block *= 0.95;

    // Non-banded kernel: ~I/~D rings only have e+1 wavefronts
    const int num_sh_wfs = (band > 0) ? (active_working_set * 3)
                                      : (active_working_set + 2 * (penalties.e + 1));
    const int max_sh_offsets_per_block = available_sh_mem_per_block / sizeof(wfa_offset_t);
    int max_sh_offsets_per_wf = max_sh_offsets_per_block / num_sh_wfs;

    // Make sure it fits in shared memory taking into account the wavefronts
    // metadata
    while ((max_sh_offsets_per_wf * sizeof(wfa_offset_t) * num_sh_wfs)
                > available_sh_mem_per_block) {
        max_sh_offsets_per_wf--;
    }

    if (band > 0) {
        // TODO: When reducing threads per block re-ca
        if (max_sh_offsets_per_wf < (threads_per_block)) {
            LOG_ERROR("There is not enough shared memory for this configuration.  Aborting.");
            exit(-1);
        } else if (max_sh_offsets_per_wf > (threads_per_block)) {
            max_sh_offsets_per_wf = threads_per_block;
        }
    } else {
        // Make it an odd number
        if ((max_sh_offsets_per_wf % 2) == 0) {
            max_sh_offsets_per_wf--;
        }
        // At least diagonal 0 in shared memory
        if (max_sh_offsets_per_wf < 1) max_sh_offsets_per_wf = 1;
    }

    // Add offsets size to shared memory
    sh_mem_size += max_sh_offsets_per_wf * sizeof(wfa_offset_t) * num_sh_wfs;

    uint32_t* next_alignment_idx = (uint32_t*)(wf_data_buffer
                                           + wf_data_buffer_size(
                                               penalties,
                                               max_steps,
                                               band
                                           ) * num_blocks);
    dim3 gridSize(num_blocks);
    dim3 blockSize(threads_per_block);

    LOG_DEBUG("Launching %d blocks of %d threads with %.2fKiB of shared memory",
              gridSize.x, blockSize.x, (float(sh_mem_size) / (1 << 10)));

    LOG_DEBUG("Working with penalties: X=%d, O=%d, E=%d", penalties.x,
              penalties.o, penalties.e);

    if (band > 0)
        alignment_kernel_aband<<<gridSize, blockSize, sh_mem_size, stream>>>(
                                                  packed_sequences_buffer,
                                                  sequences_metadata,
                                                  num_alignments,
                                                  max_steps,
                                                  wf_data_buffer,
                                                  penalties,
                                                  bt_offloaded_d,
                                                  bt_offloaded_results_d,
                                                  results_d,
                                                  next_alignment_idx,
                                                  max_sh_offsets_per_wf,
                                                  band);
    else
        alignment_kernel<<<gridSize, blockSize, sh_mem_size, stream>>>(
                                                  packed_sequences_buffer,
                                                  sequences_metadata,
                                                  num_alignments,
                                                  max_steps,
                                                  wf_data_buffer,
                                                  penalties,
                                                  bt_offloaded_d,
                                                  bt_offloaded_results_d,
                                                  results_d,
                                                  next_alignment_idx,
                                                  max_sh_offsets_per_wf);
    CUDA_CHECK_ERR
}

void copyInResults (alignment_result_t* const results,
                    const alignment_result_t* const results_d,
                    wfa_backtrace_t* const backtraces,
                    const wfa_backtrace_t* const bt_offloaded_d,
                    const size_t num_alignments,
                    const int max_steps,
                    const int num_blocks,
                    cudaStream_t stream) {
    const size_t bt_offloaded_results_size = BT_OFFLOADED_RESULT_ELEMENTS(max_steps)
                                             * num_alignments;
    const size_t bt_offloaded_size = BT_OFFLOADED_ELEMENTS(max_steps) * (size_t)num_blocks;

    const wfa_backtrace_t* const bt_offloaded_results_d = bt_offloaded_d
                                              + bt_offloaded_size;

    cudaMemcpyAsync(results, results_d, num_alignments * sizeof(alignment_result_t),
               cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_ERR
    cudaMemcpyAsync(backtraces, bt_offloaded_results_d,
               bt_offloaded_results_size * sizeof(wfa_backtrace_t),
               cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_ERR
}

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
                                       size_t available_sh_mem_per_block,
                                       int band,
                                       cudaStream_t stream) {
    const int max_wf_size = 2 * max_steps + 1;
    const int active_working_set = max(penalties.o+penalties.e, penalties.x) + 1;
    int offsets_elements = active_working_set * max_wf_size;
    offsets_elements = offsets_elements + (4 - (offsets_elements % 4));

    size_t sh_mem_size;
    if (band > 0)
        sh_mem_size = (active_working_set * sizeof(wfa_distance_aband_wavefront_t) * 3);
    else
        sh_mem_size = (active_working_set * sizeof(wfa_distance_wavefront_t) * 3);

    available_sh_mem_per_block -= sh_mem_size;
    // Using 100% of the shared memory available can give some problems, as the
    // driver sometimes take some sh memory space (?)
    available_sh_mem_per_block *= 0.95;

    const int max_sh_offsets_per_block = available_sh_mem_per_block / sizeof(wfa_offset_t);
    int max_sh_offsets_per_wf = max_sh_offsets_per_block / (active_working_set * 3);

    // Make sure if fits in shared memory taking into account the wavefronts
    // metadata
    while ((max_sh_offsets_per_wf * sizeof(wfa_offset_t) * active_working_set * 3)
                > available_sh_mem_per_block) {
        max_sh_offsets_per_wf -= 2;
    }

    if (band > 0) {
        if (max_sh_offsets_per_wf < (threads_per_block)) {
            LOG_ERROR("TODO: Less offsets than threads!!");
            exit(-1);
        } else if (max_sh_offsets_per_wf > (threads_per_block)) {
            max_sh_offsets_per_wf = threads_per_block;
        }
    } else {
        // Make it an odd number
        if ((max_sh_offsets_per_wf % 2) == 0) {
            max_sh_offsets_per_wf--;
        }
    }

    // Add offsets size to shared memory
    sh_mem_size += max_sh_offsets_per_wf * sizeof(wfa_offset_t) * active_working_set * 3;

    LOG_DEBUG("Each wavefront have %d offsets on shared memory", max_sh_offsets_per_wf)

    uint32_t* next_alignment_idx;

    if (band <= 0) {
        next_alignment_idx = (uint32_t*)(wf_data_buffer
                                               + wf_data_buffer_size_distance(
                                                   penalties,
                                                   max_steps
                                               ) * num_blocks);
    } else {
        next_alignment_idx = (uint32_t*)wf_data_buffer;

    }

    dim3 gridSize(num_blocks);
    dim3 blockSize(threads_per_block);

    LOG_DEBUG("Launching %d blocks of %d threads with %.2fKiB of shared memory",
              gridSize.x, blockSize.x, (float(sh_mem_size) / (1 << 10)));

    LOG_DEBUG("Working with penalties: X=%d, O=%d, E=%d", penalties.x,
              penalties.o, penalties.e);

    if (band > 0)
        distance_kernel_aband<<<gridSize, blockSize, sh_mem_size, stream>>>(
                                                  packed_sequences_buffer,
                                                  sequences_metadata,
                                                  num_alignments,
                                                  max_steps,
                                                  wf_data_buffer,
                                                  penalties,
                                                  results_d,
                                                  next_alignment_idx,
                                                  max_sh_offsets_per_wf,
                                                  band);
    else
        distance_kernel<<<gridSize, blockSize, sh_mem_size, stream>>>(
                                                  packed_sequences_buffer,
                                                  sequences_metadata,
                                                  num_alignments,
                                                  max_steps,
                                                  wf_data_buffer,
                                                  penalties,
                                                  results_d,
                                                  next_alignment_idx,
                                                  max_sh_offsets_per_wf);
    CUDA_CHECK_ERR
}

void copyInResults_distance (alignment_result_t* const results,
                             const alignment_result_t* const results_d,
                             const size_t num_alignments,
                             cudaStream_t stream) {

    cudaMemcpyAsync(results, results_d, num_alignments * sizeof(alignment_result_t),
               cudaMemcpyDeviceToHost, stream);
    CUDA_CHECK_ERR
}
