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

#include <stdbool.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "sequence_alignment_kernel.cuh"
#include "common_alignment_kernels.cuh"

// At least one of the highest two bits is set
//#define BT_WORD_FULL_CMP 0x40000000
#define BT_WORD_FULL_CMP (1L << (wfa_backtrace_bits-2))
#define BT_IS_FULL(bt_word) ((bt_word) >= BT_WORD_FULL_CMP)

__device__ static bt_prev_t offload_backtrace (unsigned int* const last_free_bt_position,
                                       const bt_vector_t backtrace_vector,
                                       const bt_prev_t backtrace_pointer,
                                       wfa_backtrace_t* const global_backtraces_array) {
    const bt_prev_t old_val = atomicAdd(last_free_bt_position, 1);

    // Single 64-bit store of {backtrace, prev}
    const uint2 packed = make_uint2(backtrace_vector, backtrace_pointer);
#if __CUDA_ARCH__ >= 800
    __stwt((uint2*)&global_backtraces_array[old_val], packed);
#else
    *(uint2*)&global_backtraces_array[old_val] = packed;
#endif

    // TODO: Check if new_val is more than 32 bits
    return old_val;
}

__forceinline__
__device__ static wfa_offset_t get_offset (const wfa_wavefront_t* const wf,
                                    int k,
                                    const size_t half_num_sh_offsets_per_wf) {
    const int limit = half_num_sh_offsets_per_wf;
    const int is_global = (k > limit) || (k < -limit);
    // predicate k operations
    k += limit * (k < -limit);
    k -= limit * (k > limit);
    // offsets[0] -> ptr to shared memory
    // offsets[1] -> ptr to global memory
    return wf->offsets[is_global][k];
}

__forceinline__
__device__ static void set_offset (wfa_wavefront_t* const wf,
                            int k,
                            const size_t half_num_sh_offsets_per_wf,
                            wfa_offset_t value) {
    const int limit = half_num_sh_offsets_per_wf;
    const int is_global = (k > limit) || (k < -limit);
    // predicate k operations
    k += limit * (k < -limit);
    k -= limit * (k > limit);
    // offsets[0] -> ptr to shared memory
    // offsets[1] -> ptr to global memory
    wf->offsets[is_global][k] = value;
}

__device__ __forceinline__ static uint2 ld_bt (const wfa_backtrace_t* p) {
    return *(const uint2*)p;
}

__device__ __forceinline__ static void st_bt (wfa_backtrace_t* p, const uint2 v) {
    *(uint2*)p = v;
}

__device__ __forceinline__ static uint2 push_op (uint2 bt,
                                    const affine_op_t op,
                                    unsigned int* const last_free_bt_position,
                                    wfa_backtrace_t* const offloaded_backtraces) {
    bt.x = (bt.x << 2) | op;
    if (BT_IS_FULL(bt.x)) {
        const bt_prev_t prev = offload_backtrace(last_free_bt_position,
                                                 bt.x, bt.y,
                                                 offloaded_backtraces);
        bt.x = 0;
        bt.y = prev;
    }
    return bt;
}

__device__ static void next_M (wfa_wavefront_t* M_wavefronts,
                        const int curr_wf,
                        const int active_working_set_size,
                        const int x,
                        const char* text,
                        const char* pattern,
                        const int tlen,
                        const int plen,
                        const size_t half_num_sh_offsets_per_wf,
                        unsigned int* const last_free_bt_position,
                        wfa_backtrace_t* const offloaded_backtraces,
                        const int d) {
    // The wavefront do not grow in case of mismatch
    const wfa_wavefront_t* prev_wf = &M_wavefronts[ring_add(curr_wf, x, active_working_set_size)];

    const int hi = prev_wf->hi;
    const int lo = prev_wf->lo;

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        wfa_offset_t curr_offset = get_offset(prev_wf, k, half_num_sh_offsets_per_wf) + 1;
        uint2 bt = ld_bt(&prev_wf->bt[k]);

        // Only extend and update backtrace if the previous offset exist
        if (curr_offset >= 0) {
            curr_offset = WF_extend_kernel(text, pattern,
                                           tlen, plen, k, curr_offset);
            bt = push_op(bt, OP_SUB, last_free_bt_position, offloaded_backtraces);
        }

        set_offset(&M_wavefronts[curr_wf], k, half_num_sh_offsets_per_wf, curr_offset);
        st_bt(&M_wavefronts[curr_wf].bt[k], bt);
    }

    if (threadIdx.x == 0) {
        M_wavefronts[curr_wf].hi = hi;
        M_wavefronts[curr_wf].lo = lo;
        M_wavefronts[curr_wf].exist= true;
    }
}

__device__ static void next_MDI (wfa_wavefront_t* M_wavefronts,
                          wfa_wavefront_t* I_wavefronts,
                          wfa_wavefront_t* D_wavefronts,
                          const int curr_wf,
                          const int active_working_set_size,
                          const int curr_id,
                          const int ws_id,
                          const int x,
                          const int o,
                          const int e,
                          const char* text,
                          const char* pattern,
                          const int tlen,
                          const int plen,
                          const size_t half_num_sh_offsets_per_wf,
                          unsigned int* const last_free_bt_position,
                          wfa_backtrace_t* const offloaded_backtraces,
                          const int d) {
    wfa_wavefront_t* const prev_wf_x  = &M_wavefronts[ring_add(curr_wf, x, active_working_set_size)];
    wfa_wavefront_t* const prev_wf_o  = &M_wavefronts[ring_add(curr_wf, o + e, active_working_set_size)];
    // ~I/~D only need the wavefront at s-e: their ring has e+1 slots
    wfa_wavefront_t* const prev_I_wf_e = &I_wavefronts[ring_add(curr_id, e, ws_id)];
    wfa_wavefront_t* const prev_D_wf_e = &D_wavefronts[ring_add(curr_id, e, ws_id)];

    const int hi_ID = MAX(prev_wf_o->hi, MAX(prev_I_wf_e->hi, prev_D_wf_e->hi)) + 1;
    const int hi    = MAX(prev_wf_x->hi, hi_ID);
    const int lo_ID = MIN(prev_wf_o->lo, MIN(prev_I_wf_e->lo, prev_D_wf_e->lo)) - 1;
    const int lo    = MIN(prev_wf_x->lo, lo_ID);

    for (int k=lo + threadIdx.x; k <= hi; k+=blockDim.x) {
        const wfa_offset_t I_open_off = get_offset(prev_wf_o, k - 1, half_num_sh_offsets_per_wf) + 1;
        const wfa_offset_t I_ext_off  = get_offset(prev_I_wf_e, k - 1, half_num_sh_offsets_per_wf) + 1;
        const wfa_offset_t D_open_off = get_offset(prev_wf_o, k + 1, half_num_sh_offsets_per_wf);
        const wfa_offset_t D_ext_off  = get_offset(prev_D_wf_e, k + 1, half_num_sh_offsets_per_wf);
        const wfa_offset_t X_off      = get_offset(prev_wf_x, k, half_num_sh_offsets_per_wf) + 1;

        // Load all predecessors' backtraces up front so the loads overlap
        const uint2 I_open_bt = ld_bt(&prev_wf_o->bt[k - 1]);
        const uint2 I_ext_bt  = ld_bt(&prev_I_wf_e->bt[k - 1]);
        const uint2 D_open_bt = ld_bt(&prev_wf_o->bt[k + 1]);
        const uint2 D_ext_bt  = ld_bt(&prev_D_wf_e->bt[k + 1]);
        const uint2 X_bt      = ld_bt(&prev_wf_x->bt[k]);

        // ~I: on ties gap extension wins over gap opening
        const bool I_is_ext = (I_ext_off >= I_open_off);
        const wfa_offset_t I_offset = I_is_ext ? I_ext_off : I_open_off;
        set_offset(&I_wavefronts[curr_id], k, half_num_sh_offsets_per_wf, I_offset);
        uint2 I_bt = make_uint2(0, 0);
        if (I_offset >= 0) {
            I_bt = I_is_ext ? I_ext_bt : I_open_bt;
            I_bt = push_op(I_bt, OP_INS, last_free_bt_position, offloaded_backtraces);
        }
        st_bt(&I_wavefronts[curr_id].bt[k], I_bt);

        // ~D
        const bool D_is_ext = (D_ext_off >= D_open_off);
        const wfa_offset_t D_offset = D_is_ext ? D_ext_off : D_open_off;
        set_offset(&D_wavefronts[curr_id], k, half_num_sh_offsets_per_wf, D_offset);
        uint2 D_bt = make_uint2(0, 0);
        if (D_offset >= 0) {
            D_bt = D_is_ext ? D_ext_bt : D_open_bt;
            D_bt = push_op(D_bt, OP_DEL, last_free_bt_position, offloaded_backtraces);
        }
        st_bt(&D_wavefronts[curr_id].bt[k], D_bt);

        // ~M: on ties DEL > SUB > INS (same order as the banded kernel's MAX_PB)
        wfa_offset_t M_offset = MAX(X_off, MAX(D_offset, I_offset));
        uint2 M_bt = make_uint2(0, 0);
        if (M_offset >= 0) {
            if (D_offset == M_offset) {
                M_bt = D_bt;
            } else if (X_off == M_offset) {
                M_bt = X_bt;
            } else {
                M_bt = I_bt;
            }
            M_offset = WF_extend_kernel(text, pattern, tlen, plen, k, M_offset);
            M_bt = push_op(M_bt, OP_SUB, last_free_bt_position, offloaded_backtraces);
        }

        set_offset(&M_wavefronts[curr_wf], k, half_num_sh_offsets_per_wf,  M_offset);
        st_bt(&M_wavefronts[curr_wf].bt[k], M_bt);
    }

    if (threadIdx.x == 0) {
        M_wavefronts[curr_wf].hi = hi;
        M_wavefronts[curr_wf].lo = lo;
        M_wavefronts[curr_wf].exist = true;

        I_wavefronts[curr_id].hi = hi_ID;
        I_wavefronts[curr_id].lo = lo_ID;

        D_wavefronts[curr_id].hi = hi_ID;
        D_wavefronts[curr_id].lo = lo_ID;
    }
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
                            const size_t num_sh_offsets_per_wf) {
    const int tid = threadIdx.x;
    // m = 0 for WFA
    const int x = penalties.x;
    const int o = penalties.o;
    const int e = penalties.e;

    __shared__ uint32_t alignment_idx;

    // Get the first alignment to compute from the alignments pool
    if (tid == 0) {
        alignment_idx = get_alignment_idx(next_alignment_idx);
    }

    __syncthreads();

    const size_t bt_offloaded_size = BT_OFFLOADED_ELEMENTS(max_steps);
    wfa_backtrace_t* const offloaded_backtraces = \
             &offloaded_backtraces_global[blockIdx.x * bt_offloaded_size];

    size_t bt_results_size = BT_OFFLOADED_RESULT_ELEMENTS(max_steps);

    // In shared memory:
    // - Wavefronts needed to calculate current WF_s, there are 3 "pyramids" so
    //   this number of wavefront is 3 times (WF_{max(o+e, x)} --> WF_s)
    extern __shared__ char sh_mem[];

    // TODO: +1 because of the current wf?
    const int active_working_set_size = MAX(o+e, x) + 1;
    const int max_wf_size = 2 * max_steps + 1;
    const int max_wf_size_bt = 2 * max_steps + 1;

    // ~I and ~D wavefronts are only read at distance s-e
    const int ws_id = e + 1;

    // Offsets and backtraces must be 32 bits aligned to avoid unaligned access
    // errors on the structs
    const int offsets_size = alignment_wf_elements(active_working_set_size, max_steps);
    const int bt_size = offsets_size;
    const int offsets_size_id = alignment_wf_elements(ws_id, max_steps);
    const int bt_size_id = offsets_size_id;

    uint8_t* curr_alignment_wf_data_buffer = wf_data_buffer
                    + (alignment_wf_data_buffer_size(penalties, max_steps) * blockIdx.x);

    wfa_offset_t* M_base = (wfa_offset_t*)curr_alignment_wf_data_buffer;
    wfa_offset_t* I_base = M_base + offsets_size;
    wfa_offset_t* D_base = I_base + offsets_size_id;

    // Backtraces {vector, prev} stored together (8 bytes per cell)
    wfa_backtrace_t* M_bt_base = (wfa_backtrace_t*)(D_base + offsets_size_id);
    wfa_backtrace_t* I_bt_base = M_bt_base + bt_size;
    wfa_backtrace_t* D_bt_base = I_bt_base + bt_size_id;

    // Wavefronts structres reside in shared
    wfa_wavefront_t* M_wavefronts = (wfa_wavefront_t*)sh_mem;
    wfa_wavefront_t* I_wavefronts = (M_wavefronts + active_working_set_size);
    wfa_wavefront_t* D_wavefronts = (I_wavefronts + ws_id);

    uint32_t* last_free_bt_position = (uint32_t*)
                                          (D_wavefronts + ws_id);

    wfa_offset_t* M_sh_offsets_base = (wfa_offset_t*)(last_free_bt_position + 1);
    wfa_offset_t* I_sh_offsets_base = M_sh_offsets_base
                                      + (num_sh_offsets_per_wf * active_working_set_size);
    wfa_offset_t* D_sh_offsets_base = I_sh_offsets_base
                                      + (num_sh_offsets_per_wf * ws_id);

    for (int i=tid; i<active_working_set_size; i+=blockDim.x) {
        M_wavefronts[i].offsets[1] = M_base + (i * max_wf_size) + (max_wf_size/2);
        M_wavefronts[i].offsets[0] = M_sh_offsets_base
                                         + (i * num_sh_offsets_per_wf)
                                         + (num_sh_offsets_per_wf/2);
        M_wavefronts[i].bt = M_bt_base + (i * max_wf_size_bt) + (max_wf_size_bt/2);
        if (i >= ws_id) continue;

        I_wavefronts[i].offsets[1] = I_base + (i * max_wf_size) + (max_wf_size/2);
        I_wavefronts[i].offsets[0] = I_sh_offsets_base
                                         + (i * num_sh_offsets_per_wf)
                                         + (num_sh_offsets_per_wf/2);
        I_wavefronts[i].bt = I_bt_base + (i * max_wf_size_bt) + (max_wf_size_bt/2);

        D_wavefronts[i].offsets[1] = D_base + (i * max_wf_size) + (max_wf_size/2);
        D_wavefronts[i].offsets[0] = D_sh_offsets_base
                                         + (i * num_sh_offsets_per_wf)
                                         + (num_sh_offsets_per_wf/2);
        D_wavefronts[i].bt = D_bt_base + (i * max_wf_size_bt) + (max_wf_size_bt/2);
    }


    // Initialise the global part of the wavefronts once per block. After each
    // alignment only the diagonals it touched are reset (see end of loop), so
    // the buffers are clean at the start of every alignment.
    for (int i=tid; i<(offsets_size + 2 * offsets_size_id); i+=blockDim.x) M_base[i] = OFFSET_NULL;
    for (int i=tid; i<(bt_size + 2 * bt_size_id); i+=blockDim.x) {
        st_bt(&M_bt_base[i], make_uint2(0, 0));
    }
    __syncthreads();

    // Iterate until there are no more alignments in the pool to compute
    while (alignment_idx < num_alignments) {
        const sequence_pair_t curr_batch_alignment_base = sequences_metadata[0];
        const size_t base_offset_packed = curr_batch_alignment_base.pattern_offset_packed;

        const sequence_pair_t metadata = sequences_metadata[alignment_idx];
        // Sequences with "N" chatacters not supported yet
        if (metadata.has_N) {
            if (tid == 0) {
                results[alignment_idx].distance = 0;
                results[alignment_idx].finished = false;
                alignment_idx = get_alignment_idx(next_alignment_idx);
            }
            __syncthreads();
            continue;
        }

        const char* text = packed_sequences_buffer + metadata.text_offset_packed - base_offset_packed;
        const char* pattern = packed_sequences_buffer + metadata.pattern_offset_packed - base_offset_packed ;
        const int tlen = metadata.text_len;
        const int plen = metadata.pattern_len;

        wfa_backtrace_t* const offloaded_backtrace_results_base = \
                 &offloaded_backtraces_results[alignment_idx * bt_results_size];

        // Start at 1 because 0 is used as NULL (no more backtraces blocks to
        // recover)
        *last_free_bt_position = 1;

        // Shared mem
        for (int i=tid; i<(num_sh_offsets_per_wf * (active_working_set_size + 2 * ws_id)); i+=blockDim.x) {
            M_sh_offsets_base[i] =  OFFSET_NULL;
        }


        for (int i=tid; i<active_working_set_size; i+=blockDim.x) {
            M_wavefronts[i].hi = 0;
            M_wavefronts[i].lo = 0;
            M_wavefronts[i].exist = false;
            if (i >= ws_id) continue;

            I_wavefronts[i].hi = 0;
            I_wavefronts[i].lo = 0;
            I_wavefronts[i].exist = false;

            D_wavefronts[i].hi = 0;
            D_wavefronts[i].lo = 0;
            D_wavefronts[i].exist = false;
        }

        __syncthreads();

        int curr_wf = 0;

        if (tid == 0) {
            wfa_offset_t initial_ext = WF_extend_kernel(
                text,
                pattern,
                tlen, plen,
                0, 0);
            //M_wavefronts[curr_wf].offsets[0] = initial_ext;
            set_offset(&M_wavefronts[curr_wf], 0, num_sh_offsets_per_wf/2, initial_ext);
            M_wavefronts[curr_wf].exist = true;
        }

        __syncthreads();

        // TODO: Change tarket K if we don't start form WF 0 (cooperative strategy)
        const int target_k = EWAVEFRONT_DIAGONAL(tlen, plen);
        const int target_k_abs = (target_k >= 0) ? target_k : -target_k;
        const wfa_offset_t target_offset = EWAVEFRONT_OFFSET(tlen, plen);

        bool finished = false;
        // Range of diagonals written during this alignment
        int touched_lo = 0, touched_hi = 0;
        // Ring slot of the current ~I/~D wavefront. The exist flag of each
        // slot records whether ~I/~D was computed at that distance.
        int curr_id = 0;

        int distance = 0;
        // steps = number of editions
        int steps = 0;
        // TODO: target_k_abs <= distance or <= steps (?)
        if (!(target_k_abs <= distance && M_wavefronts[curr_wf].exist && get_offset(&M_wavefronts[curr_wf], target_k, num_sh_offsets_per_wf/2) == target_offset)) {

            // Wavefronts are read "forward" in the rings, so the current index
            // moves backwards
            curr_wf = ring_dec(curr_wf, active_working_set_size);
            curr_id = ring_dec(curr_id, ws_id);

            distance++;
            steps++;
            __syncthreads();

            while (steps < (max_steps - 1)) {

                bool M_exist = false;
                bool GAP_exist = false;
                const int o_delta = ring_add(curr_wf, o + e, active_working_set_size);
                const int x_delta = ring_add(curr_wf, x, active_working_set_size);
                bool ID_computed = false;
                if ((distance - o - e) >= 0) {
                    // Just test ~I matrix as it will exist at the same wavefronts
                    // as ~D
                    GAP_exist = M_wavefronts[o_delta].exist ||
                                I_wavefronts[ring_add(curr_id, e, ws_id)].exist;
                }

                if (GAP_exist) {
                    M_exist = true;
                } else {
                    if ((distance - x) >= 0) {
                        M_exist = M_wavefronts[x_delta].exist;
                    } 
                }

                if (!GAP_exist && !M_exist) {
                    M_wavefronts[curr_wf].exist = false;
                    distance++;
                } else {
                    if (M_exist && !GAP_exist) {
                        next_M(M_wavefronts, curr_wf, active_working_set_size, x,
                               text, pattern, tlen, plen, num_sh_offsets_per_wf/2,
                               last_free_bt_position, offloaded_backtraces,
                               distance);
                    } else {
                        ID_computed = true;
                        next_MDI(
                            M_wavefronts, I_wavefronts, D_wavefronts,
                            curr_wf, active_working_set_size,
                            curr_id, ws_id,
                            x, o, e,
                            text, pattern, tlen, plen, num_sh_offsets_per_wf/2,
                            last_free_bt_position, offloaded_backtraces,
                            distance);

                        // Wavefront only grows if there's an operation in the ~I or
                        // ~D matrices
                        steps++;
                    }

                    // TODO: This is necessary for now, try to find a less sync
                    // version
                    __syncthreads();
                    touched_lo = MIN(touched_lo, (int)M_wavefronts[curr_wf].lo);
                    touched_hi = MAX(touched_hi, (int)M_wavefronts[curr_wf].hi);

                    // target_k must lie inside the wavefront: outside it the
                    // offsets are stale or beyond the allocated diagonals
                    if (target_k_abs <= distance && M_exist
                        && target_k >= M_wavefronts[curr_wf].lo
                        && target_k <= M_wavefronts[curr_wf].hi
                        && get_offset(&M_wavefronts[curr_wf], target_k, num_sh_offsets_per_wf/2) == target_offset) {
                        finished = true;
                        break;
                    }

                    distance++;
                }

            // Slot is reused e+1 steps later, so always overwrite the flag
            if (tid == 0) I_wavefronts[curr_id].exist = ID_computed;
            curr_wf = ring_dec(curr_wf, active_working_set_size);
            curr_id = ring_dec(curr_id, ws_id);

            __syncthreads();
            }
        } else {
            finished = true;
        }

        if  (tid == 0) {
            results[alignment_idx].distance = distance;
            results[alignment_idx].finished = finished;
            // Unfinished alignments are recomputed on the CPU. Their target_k
            // may be outside the wavefront (or the buffer), so do not read it.
            bt_vector_t bt_vector = 0;
            bt_prev_t bt_pointer = 0;
            if (finished) {
                bt_vector = M_wavefronts[curr_wf].bt[target_k].backtrace;
                bt_pointer = M_wavefronts[curr_wf].bt[target_k].prev;
            }

            results[alignment_idx].backtrace = {.backtrace = bt_vector,
                                             .prev = bt_pointer};

            // TODO: Change this hack
            wfa_backtrace_t curr_res_tmp = {.backtrace = bt_vector,
                                             .prev = bt_pointer};
            wfa_backtrace_t* curr_result = &curr_res_tmp;

            // Save the list in reversed order
            int i = 0;
            while (curr_result->prev != 0) {
                offloaded_backtrace_results_base[i] = \
                                            offloaded_backtraces[curr_result->prev];
                curr_result = &offloaded_backtraces[curr_result->prev];
                i++;
            }

            results[alignment_idx].num_bt_blocks = i;
        }
        __syncthreads();
        // Reset only the diagonals touched by this alignment
        {
            const int c_lo = MAX(touched_lo - 1, -max_steps);
            const int c_hi = MIN(touched_hi + 1, max_steps);
            const int width = c_hi - c_lo + 1;
            const int total = width * active_working_set_size;
            for (int j=tid; j<total; j+=blockDim.x) {
                const int slot = j / width;
                const int k = c_lo + (j % width);
                const int off_idx = slot * max_wf_size + (max_wf_size/2) + k;
                const int bt_idx = slot * max_wf_size_bt + (max_wf_size_bt/2) + k;
                M_base[off_idx] = OFFSET_NULL;
                st_bt(&M_bt_base[bt_idx], make_uint2(0, 0));
                if (slot < ws_id) {
                    I_base[off_idx] = OFFSET_NULL;
                    D_base[off_idx] = OFFSET_NULL;
                    st_bt(&I_bt_base[bt_idx], make_uint2(0, 0));
                    st_bt(&D_bt_base[bt_idx], make_uint2(0, 0));
                }
            }
        }
        // Get next alignment
        if (tid == 0) alignment_idx = get_alignment_idx(next_alignment_idx);
        __syncthreads();
    } // while
}
