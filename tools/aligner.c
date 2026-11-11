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

#include "utils/logger.h"
#include "utils/wf_clock.h"
#include "utils/arg_handler.h"
#include "utils/sequence_reader.h"
#include "utils/verification.h"
#include "utils/device_query.cuh"
#include "include/wfa_gpu.h"

#include <pthread.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <errno.h>
#include <string.h>

#define NUM_ARGUMENTS 15
#define NUM_CATEGORIES 3

typedef enum {
    CAT_IO,
    CAT_ALIGN,
    CAT_SYS
} menu_categories_t;

typedef enum {
    NONE,
    SEQ,
    FASTA
} wfa_file_format_t;

const char* menu_categories[] = {
    "Input/Output",      // 0
    "Alignment Options", // 1
    "System"             // 2
};

// pthread entry point: create the CUDA context in the background
static void* init_cuda_context_thread (void* arg) {
    (void)arg;
    init_cuda_context();
    return NULL;
}


int main(int argc, char** argv) {

    option_t options_arr[NUM_ARGUMENTS] = {
        {.name = "Input sequences file in .seq format",
         .description = "File containing the sequences to align in .seq format.",
         .category = CAT_IO,
         .short_arg = 'i',
         .long_arg = "input-seq",
         .required = false,
         .type = ARG_STR
         },
        {.name = "Input query file in .fasta format",
         .description = "File containing the query sequences to align (if not using a .seq file).",
         .category = CAT_IO,
         .short_arg = 'Q',
         .long_arg = "input-fasta-query",
         .required = false,
         .type = ARG_STR
         },
        {.name = "Input target file in .fasta format",
         .description = "File containing the target sequences to align (if not using a .seq file).",
         .category = CAT_IO,
         .short_arg = 'T',
         .long_arg = "input-fasta-target",
         .required = false,
         .type = ARG_STR
         },
        {.name = "Number of alignments",
         .description = "Number of alignments to read from the file (default=all"
                        " alignments)",
         .category = CAT_IO,
         .short_arg = 'n',
         .long_arg = "num-alignments",
         .required = false,
         .type = ARG_INT
         },
        {.name = "Affine penalties",
         .description = "Gap-affine penalties for the alignment, in format x,o,e",
         .category = CAT_ALIGN,
         .short_arg = 'g',
         .long_arg = "affine-penalties",
         .required = false,
         .type = ARG_STR
         },
        {.name = "Compute CIGAR",
         .description = "Compute the optimal alignment path (CIGAR) of all the alignments, otherwise, only the distance is computed.",
         .category = CAT_ALIGN,
         .short_arg = 'x',
         .long_arg = "compute-cigar",
         .required = false,
         .type = ARG_NO_VALUE
         },
        {.name = "Check",
         .description = "Check for alignment correctness",
         .category = CAT_SYS,
         .short_arg = 'c',
         .long_arg = "check",
         .required = false,
         .type = ARG_NO_VALUE
         },
        {.name = "Maximum error allowed",
         .description = "Maximum error that the kernel will be able to "
                        "compute (default = maximum possible error of first "
                        "alignment)",
         .category = CAT_ALIGN,
         .short_arg = 'e',
         .long_arg = "max-distance",
         .required = false,
         .type = ARG_INT
         },
        {.name = "Number of CUDA threads per alginment",
         .description = "Number of CUDA threads per block, each block computes"
                        " one or multiple alignment",
         .category = CAT_SYS,
         .short_arg = 't',
         .long_arg = "threads-per-block",
         .required = false,
         .type = ARG_INT
         },
        {.name = "Batch size",
         .description = "Number of alignments per batch.",
         .category = CAT_ALIGN,
         .short_arg = 'b',
         .long_arg = "batch-size",
         .required = false,
         .type = ARG_INT
         },
        {.name = "GPU workers",
         .description = "Number of blocks ('workers') to be running on the GPU.",
         .category = CAT_SYS,
         .short_arg = 'w',
         .long_arg = "workers",
         .required = false,
         .type = ARG_INT
         },
        {.name = "Banded execution",
         .description = "If this parameter is present, a banded approach is used (heuristic)."
                        "The parameter tells how many steps to wait until the band is re-centered."
                        " Use \"auto\" to use an automatically generated band.",
         .category = CAT_ALIGN,
         .short_arg = 'B',
         .long_arg = "band",
         .required = false,
         .type = ARG_INT
         },
        {.name = "Output File",
         .description = "File where alignment output is saved.",
         .category = CAT_IO,
         .short_arg = 'o',
         .long_arg = "output-file",
         .required = false,
         .type = ARG_STR
         },
        {.name = "Print",
         .description = "Print output to stderr",
         .category = CAT_IO,
         .short_arg = 'p',
         .long_arg = "print-output",
         .required = false,
         .type = ARG_NO_VALUE
         },
        {.name = "Verbose output",
         .description = "Add the query/target information on the output",
         .category = CAT_IO,
         .short_arg = 'O',
         .long_arg = "output-verbose",
         .required = false,
         .type = ARG_NO_VALUE
         },
    };

    int cuda_devices = 0;
    get_num_cuda_devices(&cuda_devices);
    if (cuda_devices == 0) {
        LOG_ERROR("No CUDA devices detected.")
        exit(-1);
    }

    int major, minor;
    get_cuda_capability(0, &major, &minor);

    char* device_name = get_cuda_dev_name(0);

    LOG_INFO("Using CUDA device \"%s\" with capability %d.%d",
             device_name, major, minor)

    free(device_name);

    options_t options = {options_arr, NUM_ARGUMENTS, menu_categories, NUM_CATEGORIES};

    bool success = parse_args(argc, argv, options);
    if (!success) {
        print_usage(options);
        printf("[Examples]\n");
        printf("\t./bin/wfa.affine.gpu -i sequences.seq -b <batch_size> -o scores.out\n");
        printf("\t./bin/wfa.affine.gpu -i sequences.seq -b <batch_size> -B auto -o scores-banded.out\n");
        printf("\t./bin/wfa.affine.gpu -Q queries.fasta -T targets.fasta -b <batch_size> -o scores.out\n");
        printf("\t./bin/wfa.affine.gpu -Q queries.fasta -T targets.fasta -b <batch_size> -x -o cigars.out\n");
        exit(1);
    }

    wfa_file_format_t file_format = NONE;
    option_t* opt_seq_file = get_option(options, 'i');
    option_t* opt_fasta_target = get_option(options, 'T');
    option_t* opt_fasta_query = get_option(options, 'Q');
    if (opt_fasta_target->parsed && opt_fasta_query->parsed)
        file_format = FASTA;
    if (opt_seq_file->parsed)
        file_format = SEQ;

    if (file_format == NONE) {
        LOG_ERROR("No input file provided.");
        return 1;
    }

    sequence_reader_t sequence_reader = {0};
    sequence_reader_fasta_t sequence_reader_fasta = {0};
    if (file_format == SEQ) {
        bool ok = init_sequence_reader(
                &sequence_reader,
                opt_seq_file->value.str_val);
        if (!ok) {
            LOG_ERROR("Error initializing the sequence reader.");
            return 1;
        }
    } else if (file_format == FASTA) {
        bool ok = init_sequence_reader_fasta(
                &sequence_reader_fasta,
                opt_fasta_target->value.str_val,
                opt_fasta_query->value.str_val);
        if (!ok) {
            LOG_ERROR("Error initializing the sequence reader.");
            return 1;
        }
    } else {
        LOG_ERROR("Invalid file format.");
        return 1;
    }

    size_t sequences_read = 0;
    option_t* opt_sequences_read = get_option(options, 'n');
    if (opt_sequences_read->parsed) {
        sequences_read = opt_sequences_read->value.int_val * 2;
    }

    bool check = get_option(options, 'c')->parsed;

    affine_penalties_t penalties = {0};
    // TODO: This is insecure but works for now, parse it better
    option_t* opt_penalties = get_option(options, 'g');
    int x = 2; int o = 3; int e = 1;
    if (opt_penalties->parsed) {
        int nelems = sscanf(opt_penalties->value.str_val, "%d,%d,%d", &x, &o, &e);
        if (nelems != 3) {
            LOG_WARN("Invalid penalties format provided. Using default penalties (0,2,3,1).");
            x = 2; o = 3; e = 1;
        }
    }

    if (x < 0) x *= -1;
    if (o < 0) o *= -1;
    if (e < 0) e *= -1;

    penalties.x = x;
    penalties.o = o;
    penalties.e = e;

    LOG_INFO("Penalties: M=0, X=%d, O=%d, E=%d.", penalties.x, penalties.o, penalties.e)

    // Create the CUDA context while the input file is read
    pthread_t ctx_thread;
    bool ctx_thread_ok = (pthread_create(&ctx_thread, NULL,
                                         init_cuda_context_thread, NULL) == 0);

    LOG_INFO("Reading sequences file...")
    CLOCK_INIT()
    CLOCK_START()

    if (file_format == SEQ) {
        if (!read_n_sequences(&sequence_reader, &sequences_read)) {
            LOG_ERROR("Error reading file: %s.", opt_seq_file->value.str_val);
            exit(1);
        }
    } else if (file_format == FASTA) {
        if (!read_n_sequences_fasta(&sequence_reader_fasta, &sequences_read)) {
            LOG_ERROR("Error reading files: %s and %s.",
                      opt_fasta_target->value.str_val,
                      opt_fasta_query->value.str_val);
            exit(1);
        }
    } else {
        LOG_ERROR("Invalid file format.");
        return 1;
    }

    CLOCK_STOP()
    CLOCK_REPORT("File read")
    if (ctx_thread_ok) pthread_join(ctx_thread, NULL);

    int max_distance;
    option_t* opt_max_distance = get_option(options, 'e');
    if (opt_max_distance->parsed) {
        max_distance = opt_max_distance->value.int_val;
        if (max_distance <= 0) {
            LOG_ERROR("Maximum error supported by the kernel must be > 0. Aborting.")
            exit(-1);
        }
    } else {
        // Assume error is about 10% between sequences, alignments that go
        // beyond this error will be offloaded to the CPU
        if (file_format == SEQ)
            max_distance = MAX(sequence_reader.sequences_metadata[0].text_len,
                               sequence_reader.sequences_metadata[0].pattern_len) * 0.1;
        else if (file_format == FASTA)
            max_distance = MAX(sequence_reader_fasta.sequences_metadata[0].text_len,
                               sequence_reader_fasta.sequences_metadata[0].pattern_len) * 0.1;

        max_distance *= MAX(x, MAX(o, e));
        if (max_distance > 8000) {
            LOG_WARN("Automatically genereated maximum error supported by the"
                     " kernel seems to be very high, to avoid running out of "
                     "memory, consider limiting the maximum error with the "
                     "'-e' argument.");
        }
        if (max_distance <= 20) max_distance = 20;
        LOG_INFO("No maximum error provided by the user, using %d", max_distance)
    }

    // Threads per block
    int threads_per_block;
    option_t* opt_tpb = get_option(options, 't');
    if (opt_tpb->parsed) {
        threads_per_block = opt_tpb->value.int_val;
        if (threads_per_block % 32) {
            LOG_WARN("CUDA devices use \"warps\" of 32 lanes, use a number of "
                     "threads multiple of 32 to better utilise GPU resources.")
        }
    } else {
        // If it is not provided by the user, use the maximum distance as a
        // hint.
        if (get_option(options, 'B')->parsed) {
            // Banded: the number of threads is also the band width, so it
            // must cover the whole 2*max_distance+1 wavefront
            const size_t max_wf_size = 2 * max_distance + 1;
            if (max_wf_size <= 128)       threads_per_block = 64;
            else if (max_wf_size <= 256)  threads_per_block = 128;
            else if (max_wf_size <= 512)  threads_per_block = 256;
            else if (max_wf_size <= 1024) threads_per_block = 512;
            else                          threads_per_block = 1024;
        } else {
            // Wavefronts are much narrower than 2*max_distance+1 in practice;
            // max_distance/4 (rounded up to a power of two) tracks the
            // measured optimum for short, medium and long reads.
            threads_per_block = 32;
            while (threads_per_block < 1024 && threads_per_block < max_distance / 4)
                threads_per_block *= 2;
        }
    }

    size_t num_alignments;
    if (file_format == SEQ)
        num_alignments = sequence_reader.num_sequences_read / 2;
    else if (file_format == FASTA)
        num_alignments = sequence_reader_fasta.num_sequences_read / 2;

    int batch_size;
    option_t* opt_batch_size = get_option(options, 'b');
    if (opt_batch_size->parsed) {
        batch_size = opt_batch_size->value.int_val;
    } else {
        LOG_WARN("Consider giving a batch size (-b BATCH_SIZE) to get better performance.")
        batch_size = num_alignments;
    }

    if (batch_size <= 0) {
        LOG_ERROR("Incorrect batch size (%d).", batch_size)
        exit(-1);
    }

    LOG_INFO("Batch size = %d.", batch_size)

    int num_blocks;
    option_t* opt_num_blocks = get_option(options, 'w');
    if (opt_num_blocks->parsed) {
        num_blocks = opt_num_blocks->value.int_val;
    } else {
        const int num_sm = get_cuda_SM_count(0);
        const int warps_per_block = threads_per_block / 32;
        // Assume we can get an occupancy of 32 warps / block
        const int blocks_per_sm = 32 / warps_per_block;
        num_blocks = num_sm * blocks_per_sm;
    }

    if (num_blocks <= 0) {
        LOG_ERROR("Incorrect number of workers (%d).", num_blocks)
        exit(-1);
    }

    LOG_INFO("Number of GPU workers = %d.", num_blocks)

    int band;
    option_t* opt_band = get_option(options, 'B');
    if (opt_band->parsed) {
        band = opt_band->value.int_val;
        if (band < 0) {
            LOG_ERROR("Band must positive (band=%d).", band)
            exit(-1);
        }
        if (band == 0) {
            // Automatic band = adjust band every 25 steps
            band = 25;
        }
        LOG_INFO("Banded execution. Band width: %d. Band re-centering every %d steps", threads_per_block, band)
    } else {
        band = -1;
    }

    LOG_INFO("Using %d threads per worker.", threads_per_block)

    wfa_alignment_result_t* results;
    // TODO: Make this a parameter
    const int cigar_len = max_distance * 5;
    if (!initialize_wfa_results(&results, num_alignments, cigar_len)) {
        LOG_ERROR("Can not initialise CIGAR buffer.")
        exit(-1);
    }

    wfa_alignment_options_t wfa_options = {0};
    wfa_options.max_error = max_distance;
    wfa_options.threads_per_block = threads_per_block;
    wfa_options.num_workers = num_blocks;
    wfa_options.band = band;
    wfa_options.batch_size = batch_size;
    wfa_options.num_alignments = num_alignments;
    wfa_options.penalties = penalties;

    bool compute_cigar = get_option(options, 'x')->parsed;
    char* sequences_buffer = NULL;
    size_t sequences_buffer_size = 0;
    sequence_pair_t* sequences_metadata = NULL;
    if (file_format == SEQ) {
        sequences_buffer = sequence_reader.sequences_buffer;
        sequences_buffer_size = sequence_reader.sequences_buffer_size;
        sequences_metadata = sequence_reader.sequences_metadata;
    } else if (file_format == FASTA) {
        sequences_buffer = sequence_reader_fasta.sequences_buffer;
        sequences_buffer_size = sequence_reader_fasta.sequences_buffer_size;
        sequences_metadata = sequence_reader_fasta.sequences_metadata;
    }
    CLOCK_START()
    if (compute_cigar) {
        launch_alignments(
            sequences_buffer,
            sequences_buffer_size,
            sequences_metadata,
            results,
            wfa_options,
            check
        );
    } else {
        launch_alignments_distance(
            sequences_buffer,
            sequences_buffer_size,
            sequences_metadata,
            results,
            wfa_options,
            check
        );
    }

    CLOCK_STOP()

    printf("Alignment computed. Wall time: %.3fs (%.3f alignments per second)\n",
           CLOCK_SECONDS, num_alignments / CLOCK_SECONDS);

    if (get_option(options, 'o')->parsed || get_option(options, 'p')->parsed) {

        if (!get_option(options, 'p')->parsed) {
            LOG_INFO("Writing output file...")
        }

        FILE* output_fp;
        if (!get_option(options, 'p')->parsed) {
            // Write results to output file
            char* output_file = get_option(options, 'o')->value.str_val;
            output_fp = fopen(output_file, "w");
            if (output_fp == NULL) {
                LOG_ERROR("Could not open file %s", output_file);
                exit(-1);
            }
        } else {
            output_fp = stderr;
        }

        bool verbose = get_option(options, 'O')->parsed;

        for (int i=0; i<num_alignments; i++) {
            size_t ppos = sequence_reader.sequences_metadata[i].pattern_offset;
            size_t tpos = sequence_reader.sequences_metadata[i].text_offset;
            char* pattern = &sequence_reader.sequences_buffer[ppos];
            char* text = &sequence_reader.sequences_buffer[tpos];
            char *cigar;
            if (compute_cigar) cigar = results[i].cigar.buffer;
            else cigar = "";
            if (verbose)
                fprintf(output_fp, "%d\t%s\t%s\t%s\n", -results[i].error, cigar, pattern, text);
            else
                fprintf(output_fp, "%d\t%s\n", -results[i].error, cigar);
        }

        if (!get_option(options, 'p')->parsed)
            fclose(output_fp);
    }

    destroy_wfa_results(results, num_alignments);
    destroy_reader(&sequence_reader);
}
