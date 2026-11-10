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

#ifndef WFA_TYPES_H
#define WFA_TYPES_H

#include "stdint.h"
#include "stdlib.h"

// The maximum sequence length is determined by the size of the offset, in this
// case a signed 16-bit integer (so 2^15 useful bits, as values <0 are
// considered NULL)
#define MAX_SEQ_LEN (1UL << 15)
typedef int16_t wfa_offset_t;

#define wfa_backtrace_bits 32
typedef uint32_t bt_vector_t;
typedef uint32_t bt_prev_t;

typedef struct
{
    bt_vector_t backtrace;
    bt_prev_t prev;
} wfa_backtrace_t;

typedef enum {
    OP_NOOP = 0,
    OP_INS = 1,
    OP_SUB = 2,
    OP_DEL = 3
} affine_op_t;

static const char ops_ascii[4] = {'?', 'I', 'X', 'D'};

typedef enum {
    GAP_OPEN = 1,
    GAP_EXTEND
} gap_op_t;

// Height * width
// Evaluated in size_t: the per-block count fits in 32 bits (it is checked
// against the width of bt_prev_t at allocation time), but the total for all
// blocks (num_blocks * BT_OFFLOADED_ELEMENTS) can exceed INT_MAX on devices
// with many SMs, so callers must not do this arithmetic in int.
#define BT_OFFLOADED_ELEMENTS(max_steps) \
                        (((size_t)(max_steps) * 2 + 1) \
                        * ((size_t)(max_steps) * 2 / (wfa_backtrace_bits / 2)))

#define BT_OFFLOADED_RESULT_ELEMENTS(max_steps) \
                        ((size_t)(max_steps) * 2 / (wfa_backtrace_bits / 2))

#endif
