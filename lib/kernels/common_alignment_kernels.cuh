/*
 * Copyright (c) 2022 Quim Aguado
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


#define MAX_PB(A, B) llmax((A), (B))
#define MAX(A, B) max((A), (B))
#define MIN(A, B) min((A), (B))

#define OFFSET_NULL -32000

// (a + b) mod n for 0 <= a < n and 0 <= b < n, without an integer division
__device__ __forceinline__ static int ring_add (const int a, const int b, const int n) {
    const int r = a + b;
    return (r >= n) ? (r - n) : r;
}

// (a - 1) mod n for 0 <= a < n
__device__ __forceinline__ static int ring_dec (const int a, const int n) {
    return (a == 0) ? (n - 1) : (a - 1);
}

__device__ static wfa_offset_t WF_extend_kernel (const char* text,
                                  const char* pattern,
                                  const int tlen,
                                  const int plen,
                                  const int k,
                                  const wfa_offset_t offset_k) {
    int v  = EWAVEFRONT_V(k, offset_k);
    int h  = EWAVEFRONT_H(k, offset_k);

    if (offset_k < 0 || v > plen || h > tlen) return OFFSET_NULL;

    const int bases_to_cmp = 16;
    int acc = 0;
    // Compare 16 bases at once
    while (v < plen && h < tlen) {
        // Which byte to pick
        const int real_v = v / 4;
        const int real_h = h / 4;

        // Get the displacement inside the aligned word
        const int pattern_displacement = v % bases_to_cmp;
        const int text_displacement = h % bases_to_cmp;

        // 0xffffffffffffff00
        constexpr uintptr_t alignment_mask = (uintptr_t)-1 << 2;
        uint32_t* word_p_ptr = (uint32_t*)((uintptr_t)(pattern + real_v) & alignment_mask);
        uint32_t* next_word_p_ptr = word_p_ptr + 1;
        uint32_t* word_t_ptr = (uint32_t*)((uintptr_t)(text + real_h) & alignment_mask);
        uint32_t* next_word_t_ptr = word_t_ptr + 1;

        // * 2 because each element is 2 bits
        uint32_t sub_word_p_1 = *word_p_ptr;
        uint32_t sub_word_p_2 = *next_word_p_ptr;
        sub_word_p_1 = sub_word_p_1 << (pattern_displacement * 2);
        // Convert the u32 to big-endian, as little endian inverts the order
        // for the sequences.
        sub_word_p_2 = *next_word_p_ptr;
        // Cast to uint64_t is done to avoid undefined behaviour in case
        // it's shifted by 32 elements.
        // ----
        // The type of the result is that of the promoted left operand. The
        // behavior is undefined if the right operand is negative, or
        // greater than or equal to the length in bits of the promoted left
        // operand.
        // ----
        sub_word_p_2 = ((uint64_t)sub_word_p_2) >>
            ((bases_to_cmp - pattern_displacement) * 2);

        uint32_t sub_word_t_1 = *word_t_ptr;
        sub_word_t_1 = sub_word_t_1 << (text_displacement * 2);
        uint32_t sub_word_t_2 = *next_word_t_ptr;
        sub_word_t_2 = ((uint64_t)sub_word_t_2) >>
            ((bases_to_cmp - text_displacement) * 2);

        const uint32_t word_p = sub_word_p_1 | sub_word_p_2;
        const uint32_t word_t = sub_word_t_1 | sub_word_t_2;

        uint32_t diff = word_p ^ word_t;
        // Branchless method to remove the equal bits if we read "too far away"
        constexpr uint32_t full_mask = (uint32_t)-1;
        const int next_v = v + bases_to_cmp;
        const int next_h = h + bases_to_cmp;
        const uint32_t mask_p = full_mask << ((next_v - plen) * 2 * (next_v > plen));
        const uint32_t mask_t = full_mask << ((next_h - tlen) * 2 * (next_h > tlen));
        diff = diff | ~mask_p | ~mask_t;

        int lz = __clz(diff);

        // each element has 2 bits
        const int eq_elements = lz / 2;
        acc += eq_elements;

        if (eq_elements < bases_to_cmp) {
            break;
        }


        v += bases_to_cmp;
        h += bases_to_cmp;
    }

    return offset_k + acc;
}

__forceinline__
__device__ int compute_distance_to_target (const wfa_offset_t offset,
                                          const int k,
                                          const int plen,
                                          const int tlen) {
    const wfa_offset_t left_v = plen - EWAVEFRONT_V(k, offset);
    const wfa_offset_t left_h = tlen - EWAVEFRONT_H(k, offset);
    return (offset >= 0) ? MAX(left_v,left_h) : INT_MAX;
}

__device__ static uint32_t get_alignment_idx (uint32_t* const next_alignment_idx) {
    uint32_t old_val = atomicAdd(next_alignment_idx, 1);
    return old_val;
}
