#ifndef __internal_config_hpp__
#define __internal_config_hpp__

/*
 * Copyright (c) 2016, Anonymous Institution.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#define BIG_ITER         1000000
#define MAX_CLOCK        BIG_ITER
#define INITIAL_DATA_AGE -BIG_ITER
#define INITIAL_CLOCK    0
#define BIG_STALENESS    2 * BIG_ITER
#define UPDATE_ALL_COLUMNS (0xFFFFFFFF)
#define MAX_CHANNEL      (4)
#define MAX_DC           (11)
/* The staleness to guarantee a local read.
 * We assume we won't have more than BIG_ITER iterations, and
 * the initial value for cache age is -BIG_ITER, so if we use
 * 2*BIG_ITER as the staleness, we can guarantee that we don't
 * need to fetch data from the server.
 */

#define INCREMENT_TIMING_FREQ 1
#define READ_TIMING_FREQ      1
#define SET_ROW_TIMING_FREQ   1
// #define INCREMENT_TIMING_FREQ 1000
// #define READ_TIMING_FREQ      1000
// #define SET_ROW_TIMING_FREQ   1000

#define ITERATE_CMD      1
#define OPSEQ_CMD        1

#define OP_BUFFER_SIZE   10

#define DEBUG_READ_TIME (0)

struct ParamTableRowOffset {
  int global_param_id;
  uint table_id;
  size_t start_row_id;
  size_t val_offset;
  size_t val_count;
};

#endif  // defined __internal_config_hpp__
