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

#include "align.cuh"
#include "sequence_packing.cuh"
#include "sequence_alignment.cuh"
#include "utils/cuda_utils.cuh"
#include "utils/logger.h"
#include "utils/wf_clock.h"
#include "utils/verification.cuh"
#include "utils/wfa_cpu.h"
#include "utils/cigar.h"

size_t bytes_to_copy_unpacked (const int from,
                               const int to,
                               const sequence_pair_t* sequences_metadata) {
    // +1 for the final byte
    size_t final_byte = sequences_metadata[to].text_offset +
                                 sequences_metadata[to].text_len + 1;
    size_t initial_byte = sequences_metadata[from].pattern_offset;
    return final_byte - initial_byte;
}

void launch_alignments (char* sequences_buffer,
                        const size_t sequences_buffer_size,
                        sequence_pair_t* const sequences_metadata,
                        wfa_alignment_result_t* const alignment_results,
                        wfa_alignment_options_t options,
                        bool check_correctness) {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    const size_t num_alignments = options.num_alignments;
    const affine_penalties_t penalties = options.penalties;
    const int max_distance = options.max_error;
    const int threads_per_block = options.threads_per_block;
    const int num_blocks = options.num_workers;
    size_t batch_size = options.batch_size;
    const int band = options.band;

    if (batch_size == 0) batch_size = num_alignments;
    if (batch_size > num_alignments) batch_size = num_alignments;

    int num_batchs = ceil((double)num_alignments / batch_size);

    cudaStream_t stream1, stream2;

    cudaStreamCreate(&stream1);
    CUDA_CHECK_ERR
    cudaStreamCreate(&stream2);
    CUDA_CHECK_ERR


    alignment_result_t* results;
    cudaMallocHost(&results, batch_size * sizeof(alignment_result_t));
    uint32_t backtraces_offloaded_elements = BT_OFFLOADED_RESULT_ELEMENTS(max_distance);
    wfa_backtrace_t* backtraces;
    cudaMallocHost(&backtraces, backtraces_offloaded_elements * batch_size *
                                                    sizeof(wfa_backtrace_t)
                                                    );
    CUDA_CHECK_ERR

    // TODO: Now its assumed that all the sequences will have more or less the
    // same sequence length, this may not be the case, but with this assumption
    // is possible to have only one buffer for the sequences
    char *d_seq_buffer_unpacked;
    size_t mem_needed_unpacked = sequences_metadata[batch_size-1].text_offset +
                                 sequences_metadata[batch_size-1].text_len + 1;
    // Make the buffer 50% bigger to have some extra room in case next batch
    // have slightly bigger sequences
    mem_needed_unpacked *= 1.5;

    LOG_DEBUG("Allocating %.2f MiB to store the unpacked sequences on the device.",
              (float(mem_needed_unpacked) / (1<<20)));

    cudaMalloc(&d_seq_buffer_unpacked, mem_needed_unpacked);
    cudaError_t c_err = cudaGetLastError();
    if (c_err != cudaSuccess) {
        if (c_err == cudaErrorMemoryAllocation) {
            LOG_ERROR("Not enough memory to allocate the sequences buffer on the "
                      "device. Consider decreasing the batch size.");
        } else {
        fprintf(stderr, "Error %s: %s at %s:%d\n", cudaGetErrorName(c_err),
               cudaGetErrorString(c_err), __FILE__, __LINE__);
        }
        exit(-1);
    }

    char *d_seq_buffer_packed;
    size_t mem_needed_packed = 0;

    // Calculate the amount of memory that will be needed to store the packed
    // sequences on the GPU. Each packed sequence has to be 32 bits aligned, so
    // the corresponding padding is added.
    // Update the metadata so the offsets matches the packed sequences.
    for (int i=0; i<batch_size; i++) {
        // Pattern
        sequence_pair_t* curr_alignment = &sequences_metadata[i];
        size_t pattern_length_packed = curr_alignment->pattern_len/4;
        curr_alignment->pattern_offset_packed = mem_needed_packed;
        mem_needed_packed += (pattern_length_packed
                        + (4 - (pattern_length_packed % 4)));
        // Text
        size_t text_length_packed = curr_alignment->text_len/4;
        curr_alignment->text_offset_packed = mem_needed_packed;
        mem_needed_packed += (text_length_packed
                        + (4 - (text_length_packed % 4)));
    }

    // Make buffer 50% bigger
    mem_needed_packed *= 1.5;
    mem_needed_packed = mem_needed_packed - (4 - (mem_needed_packed % 4));

    LOG_DEBUG("Allocating %.2f MiB to store the packed sequences on the device.",
              (float(mem_needed_packed) / (1<<20)));

    cudaMalloc(&d_seq_buffer_packed, mem_needed_packed);
    c_err = cudaGetLastError();
    if (c_err != cudaSuccess) {
        if (c_err == cudaErrorMemoryAllocation) {
            LOG_ERROR("Not enough memory to allocate the sequences buffer on the "
                      "device. Consider decreasing the batch size.");
        } else {
        fprintf(stderr, "Error %s: %s at %s:%d\n", cudaGetErrorName(c_err),
               cudaGetErrorString(c_err), __FILE__, __LINE__);
        }
        exit(-1);
    }

    sequence_pair_t *d_seq_metadata;
    size_t mem_needed_metadata = batch_size * sizeof(sequence_pair_t);
    LOG_DEBUG("Allocating %.2f MiB to store the packed sequences metadata on "
              "the device.", float(mem_needed_metadata) / (1<<20));
    cudaMalloc(&d_seq_metadata, mem_needed_metadata);
    c_err = cudaGetLastError();
    if (c_err != cudaSuccess) {
        if (c_err == cudaErrorMemoryAllocation) {
            LOG_ERROR("Not enough memory to allocate the sequences metadata buffer "
                      "on the device. Consider decreasing the batch size.");
        } else {
        fprintf(stderr, "Error %s: %s at %s:%d\n", cudaGetErrorName(c_err),
               cudaGetErrorString(c_err), __FILE__, __LINE__);
        }
        exit(-1);
    }

    LOG_DEBUG("Aligning %zu alignments using %d batchs of %zu elements.",
              num_alignments, num_batchs, batch_size);

    // Copy unpacked sequences of current batch
    size_t bytes_to_copy_seqs = bytes_to_copy_unpacked(0, batch_size - 1,
                                                       sequences_metadata);
    const char *initial_seq = &sequences_buffer[sequences_metadata[0].pattern_offset];
    if ((initial_seq + bytes_to_copy_seqs - sequences_buffer) > sequences_buffer_size) {
        LOG_ERROR("Reading out of sequences buffer. Aborting.");
        return;
    }
    cudaMemcpyAsync(d_seq_buffer_unpacked, initial_seq, bytes_to_copy_seqs,
                    cudaMemcpyHostToDevice, stream1);
    CUDA_CHECK_ERR

    // Results of the alignments
    alignment_result_t *results_d;
    cudaMalloc(&results_d, batch_size * sizeof(alignment_result_t));
    c_err = cudaGetLastError();
    if (c_err != cudaSuccess) {
        if (c_err == cudaErrorMemoryAllocation) {
            LOG_ERROR("Not enough memory to allocate the results buffer on the "
                      "device. Consider decreasing the batch size.");
        } else {
        fprintf(stderr, "Error %s: %s at %s:%d\n", cudaGetErrorName(c_err),
               cudaGetErrorString(c_err), __FILE__, __LINE__);
        }
        exit(-1);
    }
    CUDA_CHECK_ERR

    // Space for the kernel to store the offloaded backtraces on the GPU, this
    // is allocated just once here and reused.
    wfa_backtrace_t* bt_offloaded_d;
    // TODO: max_distance = max_steps (?)
    allocate_offloaded_bt_d(&bt_offloaded_d, max_distance, num_blocks, batch_size);

    uint8_t* wf_data_buffer;
    allocate_wf_data_buffer_d(&wf_data_buffer, max_distance,
                              penalties, num_blocks, band);

    // Global counter of the next alignment to compute, after the working sets
    uint32_t* next_alignment_idx_d = (uint32_t*)(wf_data_buffer +
                                     wf_data_buffer_size(
                                         penalties,
                                         max_distance,
                                         band) * num_blocks
                                     );

    int from, to, curr_batch_size, batch;

    size_t available_sh_mem_per_block = available_shared_mem_per_block(
            penalties,
            max_distance,
            threads_per_block);

    for (batch=0; batch < num_batchs; batch++) {

        const int prev_from = from;
        const int prev_to = to;
        const int prev_curr_batch_size = curr_batch_size;
        from = batch * batch_size;
        to = (batch == (num_batchs-1))
                       ? num_alignments-1 : (((batch+1) * batch_size) - 1);
        curr_batch_size = to - from + 1;

        // Copy metadata of current batch
        cudaMemcpyAsync(d_seq_metadata, &sequences_metadata[from],
                        curr_batch_size * sizeof(sequence_pair_t),
                        cudaMemcpyHostToDevice, stream1);
        CUDA_CHECK_ERR

        // The kernel keeps the working set clean and never reads offloaded
        // backtraces it did not write, only the alignment counter needs reset.
        cudaMemsetAsync(next_alignment_idx_d, 0, sizeof(uint32_t), stream2);
        CUDA_CHECK_ERR


        // Wait for unpacked sequences memcpys to finish
        cudaStreamSynchronize(stream1);

        // Launch packing kernel
        pack_sequences_gpu_async(d_seq_buffer_unpacked,
                                 d_seq_buffer_packed,
                                 d_seq_metadata,
                                 curr_batch_size,
                                 stream2);

        // Make sure packing kernel have finished before sending the next
        // unpacked sequences to the device
        cudaStreamSynchronize(stream2);

        // Align current batch
        launch_alignments_async(
            d_seq_buffer_packed,
            d_seq_metadata,
            curr_batch_size,
            penalties,
            results,
            backtraces,
            results_d,
            bt_offloaded_d,
            wf_data_buffer,
            max_distance,
            threads_per_block,
            num_blocks,
            available_sh_mem_per_block,
            band,
            stream2
        );


        // Compute previous batch alignments that need to be offloaded to CPU
        // and expand CIGARS, while the current batch alignments are computed.
        if (batch > 0) {
            int alignments_computed_cpu = compute_alignments_cpu_threaded(
                    prev_curr_batch_size,
                    prev_from,
                    results,
                    alignment_results,
                    sequences_metadata,
                    sequences_buffer,
                    backtraces,
                    backtraces_offloaded_elements,
                    penalties.x, penalties.o, penalties.e,
                    // Use heuristic WFA-CPU if banded WFA-GPU is being used
                    (band > 0)
                    );

            if (alignments_computed_cpu > 0) {
                LOG_DEBUG("(Batch %d) %d/%d alignemnts could not be computed on the"
                         " GPU and where offloaded to the CPU.",
                         batch-1, alignments_computed_cpu, prev_curr_batch_size)
            }

            // Check correctness if asked
            if (check_correctness) {
                LOG_DEBUG("Checking batch %d correctnes.", batch);
                const uint32_t backtraces_offloaded_elements = BT_OFFLOADED_RESULT_ELEMENTS(max_distance);
                float avg_distance = 0;
                int correct = 0;
                int incorrect = 0;
                CLOCK_INIT()
                CLOCK_START()
                #pragma omp parallel for reduction(+:avg_distance,correct,incorrect) schedule(dynamic)
                for (int i=prev_from; i<=prev_to; i++) {
                    if (!results[i-prev_from].finished) {
                        correct++;
                        avg_distance += results[i-prev_from].distance;
                        continue;
                    }

                    size_t toffset = sequences_metadata[i].text_offset;
                    size_t poffset = sequences_metadata[i].pattern_offset;

                    const char* text = &sequences_buffer[toffset];
                    const char* pattern = &sequences_buffer[poffset];

                    size_t tlen = sequences_metadata[i].text_len;
                    size_t plen = sequences_metadata[i].pattern_len;

                    int distance = results[i-prev_from].distance;
                    char* cigar = recover_cigar(text, pattern, tlen,
                                                plen, results[i-prev_from].backtrace,
                                                backtraces + backtraces_offloaded_elements*(i-prev_from),
                                                results[i-prev_from]);


                    bool correct_cigar = check_cigar_edit(text, pattern, tlen, plen, cigar);
                    bool correct_affine_d = check_affine_distance(text, pattern, tlen,
                                                                  plen, distance,
                                                                  penalties, cigar);

                    if (!correct_cigar) {
                        LOG_ERROR("Incorrect cigar %d (%d). Distance: %d. CIGAR: %s\n", i-prev_from, i, distance, cigar);
                    }


                    int cpu_computed_distance = compute_alignment_cpu(
                        pattern, text,
                        plen, tlen,
                        penalties.x, penalties.o, penalties.e
                    );

                    bool gpu_distance_ok = (distance == cpu_computed_distance);
                    if (!gpu_distance_ok) {
                        LOG_ERROR("Incorrect distance (%d). GPU=%d, CPU=%d", i, distance, cpu_computed_distance)
                    }

                    avg_distance += distance;

                    if (correct_cigar && correct_affine_d && gpu_distance_ok) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                    free(cigar);
                }

                avg_distance /= prev_curr_batch_size;
                CLOCK_STOP()
                LOG_DEBUG("(Batch %d) correct=%d Incorrect=%d Average score=%f (%.3f"
                         " alignments per second checked)\n", batch-1, correct,
                         incorrect, avg_distance, prev_curr_batch_size/ CLOCK_SECONDS);
            }
        }

        copyInResults(results,
            results_d,
            backtraces,
            bt_offloaded_d,
            curr_batch_size,
            max_distance,
            num_blocks,
            stream2);

        // Copy unpacked sequences of next batch while the alignment kernel is
        // running. Don't do it in the final batch as there is no next batch in
        // that case.
        if (batch < (num_batchs-1)) {
            const int next_from = (batch+1) * batch_size;
            const int next_to = ((batch+1) == (num_batchs-1))
                           ? num_alignments-1 : (((batch+2) * batch_size) - 1);
            bytes_to_copy_seqs = bytes_to_copy_unpacked(next_from, next_to,
                                                        sequences_metadata);
            if (bytes_to_copy_seqs > mem_needed_unpacked) {
                LOG_ERROR("Sequences buffer is too small to fit the current batch. Aborting.");
                break;
            }

            initial_seq = &sequences_buffer[sequences_metadata[next_from].pattern_offset];
            if ((initial_seq + bytes_to_copy_seqs - sequences_buffer) > sequences_buffer_size) {
                LOG_ERROR("Reading out of sequences buffer. Aborting.");
                break;
            }
            cudaMemcpyAsync(d_seq_buffer_unpacked, initial_seq, bytes_to_copy_seqs,
                            cudaMemcpyHostToDevice, stream1);
            CUDA_CHECK_ERR

            size_t next_batch_size = next_to - next_from + 1;

            mem_needed_packed = 0;
            // Prepare metadata for next batch (calculate)
            for (int i=0; i<next_batch_size; i++) {
                // Pattern
                sequence_pair_t* curr_alignment = &sequences_metadata[next_from + i];
                size_t pattern_length_packed = curr_alignment->pattern_len/4;
                curr_alignment->pattern_offset_packed = mem_needed_packed;
                mem_needed_packed += (pattern_length_packed
                                + (4 - (pattern_length_packed % 4)));
                // Text
                size_t text_length_packed = curr_alignment->text_len/4;
                curr_alignment->text_offset_packed = mem_needed_packed;
                mem_needed_packed += (text_length_packed
                                + (4 - (text_length_packed % 4)));
            }
        }

        // Make sure that alignment kernel has finished before packing the next
        // batch sequences
        cudaStreamSynchronize(stream2);

        LOG_DEBUG("Batch %d/%d computed", batch+1, num_batchs);
    }

    // Compute alignments and expand CIGARS of the last batch that need to be
    // computed on CPU
    int alignments_computed_cpu = compute_alignments_cpu_threaded(
            curr_batch_size,
            from,
            results,
            alignment_results,
            sequences_metadata,
            sequences_buffer,
            backtraces,
            backtraces_offloaded_elements,
            penalties.x, penalties.o, penalties.e,
            // Use heuristic WFA-CPU if banded WFA-GPU is being used
            (band > 0)
            );

    if (alignments_computed_cpu > 0) {
        LOG_DEBUG("(Batch %d) %d/%d alignemnts could not be computed on the"
                 " GPU and where offloaded to the CPU.",
                 batch, alignments_computed_cpu, curr_batch_size)
    }

    // Check correctness if asked
    if (check_correctness) {
        LOG_DEBUG("Checking batch %d correctnes.", batch+1);
        const uint32_t backtraces_offloaded_elements = BT_OFFLOADED_RESULT_ELEMENTS(max_distance);
        float avg_distance = 0;
        int correct = 0;
        int incorrect = 0;
        CLOCK_INIT()
        CLOCK_START()
        #pragma omp parallel for reduction(+:avg_distance,correct,incorrect)
        for (int i=from; i<=to; i++) {
            if (!results[i-from].finished) {
                correct++;
                avg_distance += results[i-from].distance;
                continue;
            }

            size_t toffset = sequences_metadata[i].text_offset;
            size_t poffset = sequences_metadata[i].pattern_offset;

            const char* text = &sequences_buffer[toffset];
            const char* pattern = &sequences_buffer[poffset];

            size_t tlen = sequences_metadata[i].text_len;
            size_t plen = sequences_metadata[i].pattern_len;

            int distance = results[i-from].distance;
            char* cigar = recover_cigar(text, pattern, tlen,
                                        plen, results[i-from].backtrace,
                                        backtraces + backtraces_offloaded_elements*(i-from),
                                        results[i-from]);


            bool correct_cigar = check_cigar_edit(text, pattern, tlen, plen, cigar);
            bool correct_affine_d = check_affine_distance(text, pattern, tlen,
                                                          plen, distance,
                                                          penalties, cigar);

            if (!correct_cigar) {
                LOG_ERROR("Incorrect cigar %d (%d). Distance: %d. CIGAR: %s\n", i-from, i, distance, cigar);
            }

            avg_distance += distance;

            if (correct_cigar && correct_affine_d) {
                correct++;
            } else {
                incorrect++;
            }
            free(cigar);
        }

        avg_distance /= curr_batch_size;
        CLOCK_STOP()
        LOG_DEBUG("(Batch %d) correct=%d Incorrect=%d Average score=%f (%.3f"
                 " alignments per second checked)\n", batch, correct,
                 incorrect, avg_distance, curr_batch_size/ CLOCK_SECONDS);
    }

    cudaStreamDestroy(stream1);
    CUDA_CHECK_ERR
    cudaStreamDestroy(stream2);
    CUDA_CHECK_ERR

    cudaFree(d_seq_buffer_unpacked);
    cudaFree(d_seq_buffer_packed);
    cudaFree(d_seq_metadata);
    cudaFree(results_d);
    cudaFree(bt_offloaded_d);
    cudaFree(wf_data_buffer);
    cudaFreeHost(results);
    cudaFreeHost(backtraces);
}


void launch_alignments_distance (char* sequences_buffer,
                                 const size_t sequences_buffer_size,
                                 sequence_pair_t* const sequences_metadata,
                                 wfa_alignment_result_t* const alignment_results,
                                 wfa_alignment_options_t options,
                                 bool check_correctness) {
    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);

    const size_t num_alignments = options.num_alignments;
    const affine_penalties_t penalties = options.penalties;
    const int max_distance = options.max_error;
    const int threads_per_block = options.threads_per_block;
    const int num_blocks = options.num_workers;
    size_t batch_size = options.batch_size;
    const int band = options.band;

    if (batch_size == 0) batch_size = num_alignments;
    if (batch_size > num_alignments) batch_size = num_alignments;

    int num_batchs = ceil((double)num_alignments / batch_size);

    cudaStream_t stream1, stream2;

    cudaStreamCreate(&stream1);
    CUDA_CHECK_ERR
    cudaStreamCreate(&stream2);
    CUDA_CHECK_ERR


    alignment_result_t* results;
    cudaMallocHost(&results, batch_size * sizeof(alignment_result_t));
    CUDA_CHECK_ERR

    // TODO: Now its assumed that all the sequences will have more or less the
    // same sequence length, this may not be the case, but with this assumption
    // is possible to have only one buffer for the sequences
    char *d_seq_buffer_unpacked;
    size_t mem_needed_unpacked = sequences_metadata[batch_size-1].text_offset +
                                 sequences_metadata[batch_size-1].text_len + 1;
    // Make the buffer 50% bigger to have some extra room in case next batch
    // have slightly bigger sequences
    mem_needed_unpacked *= 1.5;

    LOG_DEBUG("Allocating %.2f MiB to store the unpacked sequences on the device.",
              (float(mem_needed_unpacked) / (1<<20)));

    cudaMalloc(&d_seq_buffer_unpacked, mem_needed_unpacked);
    cudaError_t c_err = cudaGetLastError();
    if (c_err != cudaSuccess) {
        if (c_err == cudaErrorMemoryAllocation) {
            LOG_ERROR("Not enough memory to allocate the sequences buffer on the "
                      "device. Consider decreasing the batch size.");
        } else {
        fprintf(stderr, "Error %s: %s at %s:%d\n", cudaGetErrorName(c_err),
               cudaGetErrorString(c_err), __FILE__, __LINE__);
        }
        exit(-1);
    }

    char *d_seq_buffer_packed;
    size_t mem_needed_packed = 0;

    // Calculate the amount of memory that will be needed to store the packed
    // sequences on the GPU. Each packed sequence has to be 32 bits aligned, so
    // the corresponding padding is added.
    // Update the metadata so the offsets matches the packed sequences.
    for (int i=0; i<batch_size; i++) {
        // Pattern
        sequence_pair_t* curr_alignment = &sequences_metadata[i];
        size_t pattern_length_packed = curr_alignment->pattern_len/4;
        curr_alignment->pattern_offset_packed = mem_needed_packed;
        mem_needed_packed += (pattern_length_packed
                        + (4 - (pattern_length_packed % 4)));
        // Text
        size_t text_length_packed = curr_alignment->text_len/4;
        curr_alignment->text_offset_packed = mem_needed_packed;
        mem_needed_packed += (text_length_packed
                        + (4 - (text_length_packed % 4)));
    }

    // Make buffer 50% bigger
    mem_needed_packed *= 1.5;
    mem_needed_packed = mem_needed_packed - (4 - (mem_needed_packed % 4));

    LOG_DEBUG("Allocating %.2f MiB to store the packed sequences on the device.",
              (float(mem_needed_packed) / (1<<20)));

    cudaMalloc(&d_seq_buffer_packed, mem_needed_packed);
    c_err = cudaGetLastError();
    if (c_err != cudaSuccess) {
        if (c_err == cudaErrorMemoryAllocation) {
            LOG_ERROR("Not enough memory to allocate the sequences buffer on the "
                      "device. Consider decreasing the batch size.");
        } else {
        fprintf(stderr, "Error %s: %s at %s:%d\n", cudaGetErrorName(c_err),
               cudaGetErrorString(c_err), __FILE__, __LINE__);
        }
        exit(-1);
    }

    sequence_pair_t *d_seq_metadata;
    size_t mem_needed_metadata = batch_size * sizeof(sequence_pair_t);
    LOG_DEBUG("Allocating %.2f MiB to store the packed sequences metadata on "
              "the device.", float(mem_needed_metadata) / (1<<20));
    cudaMalloc(&d_seq_metadata, mem_needed_metadata);
    c_err = cudaGetLastError();
    if (c_err != cudaSuccess) {
        if (c_err == cudaErrorMemoryAllocation) {
            LOG_ERROR("Not enough memory to allocate the sequence metadata buffer "
                      "on the device. Consider decreasing the batch size.");
        } else {
        fprintf(stderr, "Error %s: %s at %s:%d\n", cudaGetErrorName(c_err),
               cudaGetErrorString(c_err), __FILE__, __LINE__);
        }
        exit(-1);
    }

    LOG_DEBUG("Aligning %zu alignments using %d batchs of %zu elements.",
              num_alignments, num_batchs, batch_size);

    // Copy unpacked sequences of current batch
    size_t bytes_to_copy_seqs = bytes_to_copy_unpacked(0, batch_size - 1,
                                                       sequences_metadata);
    const char *initial_seq = &sequences_buffer[sequences_metadata[0].pattern_offset];
    if ((initial_seq + bytes_to_copy_seqs - sequences_buffer) > sequences_buffer_size) {
        LOG_ERROR("Reading out of sequences buffer. Aborting.");
        return;
    }
    cudaMemcpyAsync(d_seq_buffer_unpacked, initial_seq, bytes_to_copy_seqs,
                    cudaMemcpyHostToDevice, stream1);
    CUDA_CHECK_ERR

    // Results of the alignments
    alignment_result_t *results_d;
    cudaMalloc(&results_d, batch_size * sizeof(alignment_result_t));
    c_err = cudaGetLastError();
    if (c_err != cudaSuccess) {
        if (c_err == cudaErrorMemoryAllocation) {
            LOG_ERROR("Not enough memory to allocate the results buffer on the "
                      "device. Consider decreasing the batch size.");
        } else {
        fprintf(stderr, "Error %s: %s at %s:%d\n", cudaGetErrorName(c_err),
               cudaGetErrorString(c_err), __FILE__, __LINE__);
        }
        exit(-1);
    }

    uint8_t* wf_data_buffer = NULL;
    if (band <= 0) {
        // The banded strategy uses only shared memory, so there is no need to
        // allocate the WF buffers in global memory
        allocate_wf_data_buffer_distance_d(&wf_data_buffer, max_distance,
                                  penalties, num_blocks);
    } else {
        // Make just a small allocation for the "next_alignment_idx"
        allocate_wf_data_buffer_distance_d(&wf_data_buffer, 1, penalties, 1);
    }


    size_t available_sh_mem_per_block = available_shared_mem_per_block(
            penalties,
            max_distance,
            threads_per_block);

    int from, to, curr_batch_size, batch;

    for (batch=0; batch < num_batchs; batch++) {

        const int prev_from = from;
        const int prev_to = to;
        const int prev_curr_batch_size = curr_batch_size;
        from = batch * batch_size;
        to = (batch == (num_batchs-1))
                       ? num_alignments-1 : (((batch+1) * batch_size) - 1);
        curr_batch_size = to - from + 1;

        // Copy metadata of current batch
        cudaMemcpyAsync(d_seq_metadata, &sequences_metadata[from],
                        curr_batch_size * sizeof(sequence_pair_t),
                        cudaMemcpyHostToDevice, stream1);
        CUDA_CHECK_ERR

        // Reset the memory regions for the alignment kernel
        if (band <= 0) {
            reset_wf_data_buffer_distance_d(wf_data_buffer, max_distance,
                                            penalties, num_blocks, stream2);
        } else {
            reset_wf_data_buffer_distance_d(wf_data_buffer, 1,
                                            penalties, 1, stream2);
        }

        // Wait for unpacked sequences memcpys to finish
        cudaStreamSynchronize(stream1);

        // Launch packing kernel
        pack_sequences_gpu_async(d_seq_buffer_unpacked,
                                 d_seq_buffer_packed,
                                 d_seq_metadata,
                                 curr_batch_size,
                                 stream2);

        // Make sure packing kernel have finished before sending the next
        // unpacked sequences to the device
        cudaStreamSynchronize(stream2);

        // Align current batch
        launch_alignments_distance_async(
            d_seq_buffer_packed,
            d_seq_metadata,
            curr_batch_size,
            penalties,
            results,
            results_d,
            wf_data_buffer,
            max_distance,
            threads_per_block,
            num_blocks,
            available_sh_mem_per_block,
            band,
            stream2
        );

        // Compute previous batch alignments that need to be offloaded to CPU
        // and expand CIGARS, while the current batch alignments are computed.
        if (batch > 0) {
            int alignments_computed_cpu = compute_distance_cpu_threaded(
                    prev_curr_batch_size,
                    prev_from,
                    results,
                    alignment_results,
                    sequences_metadata,
                    sequences_buffer,
                    penalties.x, penalties.o, penalties.e,
                    // Use heuristic WFA-CPU if banded WFA-GPU is being used
                    (band > 0)
                    );

            if (alignments_computed_cpu > 0) {
                LOG_DEBUG("(Batch %d) %d/%d alignemnts could not be computed on the"
                         " GPU and where offloaded to the CPU.",
                         batch-1, alignments_computed_cpu, prev_curr_batch_size)
            }

            // Check correctness if asked
            if (check_correctness) {
                LOG_DEBUG("Checking batch %d correctnes.", batch);
                float avg_distance = 0;
                int correct = 0;
                int incorrect = 0;
                CLOCK_INIT()
                CLOCK_START()
                #pragma omp parallel for reduction(+:avg_distance,correct,incorrect) schedule(dynamic)
                for (int i=prev_from; i<=prev_to; i++) {
                    if (!results[i-prev_from].finished) {
                        correct++;
                        avg_distance += results[i-prev_from].distance;
                        continue;
                    }

                    size_t toffset = sequences_metadata[i].text_offset;
                    size_t poffset = sequences_metadata[i].pattern_offset;

                    const char* text = &sequences_buffer[toffset];
                    const char* pattern = &sequences_buffer[poffset];

                    size_t tlen = sequences_metadata[i].text_len;
                    size_t plen = sequences_metadata[i].pattern_len;

                    int distance = results[i-prev_from].distance;

                    int cpu_computed_distance = compute_alignment_cpu(
                        pattern, text,
                        plen, tlen,
                        penalties.x, penalties.o, penalties.e
                    );

                    bool gpu_distance_ok = (distance == cpu_computed_distance);
                    //if (!gpu_distance_ok) {
                    //    LOG_ERROR("Incorrect distance (%d). GPU=%d, CPU=%d", i, distance, cpu_computed_distance)
                    //}

                    avg_distance += distance;

                    if (gpu_distance_ok) {
                        correct++;
                    } else {
                        incorrect++;
                    }
                }

                avg_distance /= prev_curr_batch_size;
                CLOCK_STOP()
                LOG_INFO("(Batch %d) correct=%d Incorrect=%d Average score=%f (%.3f"
                         " alignments per second checked)\n", batch-1, correct,
                         incorrect, avg_distance, prev_curr_batch_size/ CLOCK_SECONDS);
            }
        }

        copyInResults_distance(results,
            results_d,
            curr_batch_size,
            stream2);

        // Copy unpacked sequences of next batch while the alignment kernel is
        // running. Don't do it in the final batch as there is no next batch in
        // that case.
        if (batch < (num_batchs-1)) {
            const int next_from = (batch+1) * batch_size;
            const int next_to = ((batch+1) == (num_batchs-1))
                           ? num_alignments-1 : (((batch+2) * batch_size) - 1);
            bytes_to_copy_seqs = bytes_to_copy_unpacked(next_from, next_to,
                                                        sequences_metadata);
            if (bytes_to_copy_seqs > mem_needed_unpacked) {
                LOG_ERROR("Sequences buffer is too small to fit the current batch. Aborting.");
                break;
            }

            initial_seq = &sequences_buffer[sequences_metadata[next_from].pattern_offset];
            if ((initial_seq + bytes_to_copy_seqs - sequences_buffer) > sequences_buffer_size) {
                LOG_ERROR("Reading out of sequences buffer. Aborting.");
                break;
            }
            cudaMemcpyAsync(d_seq_buffer_unpacked, initial_seq, bytes_to_copy_seqs,
                            cudaMemcpyHostToDevice, stream1);
            CUDA_CHECK_ERR

            size_t next_batch_size = next_to - next_from + 1;

            mem_needed_packed = 0;
            // Prepare metadata for next batch (calculate)
            for (int i=0; i<next_batch_size; i++) {
                // Pattern
                sequence_pair_t* curr_alignment = &sequences_metadata[next_from + i];
                size_t pattern_length_packed = curr_alignment->pattern_len/4;
                curr_alignment->pattern_offset_packed = mem_needed_packed;
                mem_needed_packed += (pattern_length_packed
                                + (4 - (pattern_length_packed % 4)));
                // Text
                size_t text_length_packed = curr_alignment->text_len/4;
                curr_alignment->text_offset_packed = mem_needed_packed;
                mem_needed_packed += (text_length_packed
                                + (4 - (text_length_packed % 4)));
            }
        }

        // Make sure that alignment kernel has finished before packing the next
        // batch sequences
        cudaStreamSynchronize(stream2);

        LOG_DEBUG("Batch %d/%d computed", batch+1, num_batchs);
    }

    // Compute alignments and expand CIGARS of the last batch that need to be
    // computed on CPU
    // TODO
    int alignments_computed_cpu = compute_distance_cpu_threaded(
            curr_batch_size,
            from,
            results,
            alignment_results,
            sequences_metadata,
            sequences_buffer,
            penalties.x, penalties.o, penalties.e,
            // Use heuristic WFA-CPU if banded WFA-GPU is being used
            (band > 0)
            );

    if (alignments_computed_cpu > 0) {
        LOG_DEBUG("(Batch %d) %d/%d alignemnts could not be computed on the"
                 " GPU and where offloaded to the CPU.",
                 batch, alignments_computed_cpu, curr_batch_size)
    }

    // Check correctness if asked
    if (check_correctness) {
        LOG_DEBUG("Checking batch %d correctnes.", batch+1);
        float avg_distance = 0;
        int correct = 0;
        int incorrect = 0;
        CLOCK_INIT()
        CLOCK_START()
        #pragma omp parallel for reduction(+:avg_distance,correct,incorrect)
        for (int i=from; i<=to; i++) {
            if (!results[i-from].finished) {
                correct++;
                avg_distance += results[i-from].distance;
                continue;
            }

            size_t toffset = sequences_metadata[i].text_offset;
            size_t poffset = sequences_metadata[i].pattern_offset;

            const char* text = &sequences_buffer[toffset];
            const char* pattern = &sequences_buffer[poffset];

            size_t tlen = sequences_metadata[i].text_len;
            size_t plen = sequences_metadata[i].pattern_len;

            int distance = results[i-from].distance;
            int cpu_computed_distance = compute_alignment_cpu(
                pattern, text,
                plen, tlen,
                penalties.x, penalties.o, penalties.e
            );

            bool gpu_distance_ok = (distance == cpu_computed_distance);
            //if (!gpu_distance_ok) {
            //    LOG_ERROR("Incorrect distance (%d). GPU=%d, CPU=%d", i, distance, cpu_computed_distance)
            //}

            avg_distance += distance;

            if (gpu_distance_ok) {
                correct++;
            } else {
                incorrect++;
            }
        }

        avg_distance /= curr_batch_size;
        CLOCK_STOP()
        LOG_INFO("(Batch %d) correct=%d Incorrect=%d Average score=%f (%.3f"
                 " alignments per second checked)\n", batch, correct,
                 incorrect, avg_distance, curr_batch_size/ CLOCK_SECONDS);
    }

    cudaStreamDestroy(stream1);
    CUDA_CHECK_ERR
    cudaStreamDestroy(stream2);
    CUDA_CHECK_ERR

    cudaFree(d_seq_buffer_unpacked);
    cudaFree(d_seq_buffer_packed);
    cudaFree(d_seq_metadata);
    cudaFree(results_d);
    cudaFree(wf_data_buffer);
    cudaFreeHost(results);
}
