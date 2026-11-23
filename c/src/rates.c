/* SPDX-License-Identifier: Apache-2.0 */
/* Frame lengths and the decoder resampling ratio of TS 103 190-1 Tables
 * 83 (48 kHz) and 84 (44,1 kHz). Port of ac4/rates.py. */
#include "ac4_internal.h"

/* frame_rate_index -> frame_len_base at 48 kHz, and the ratio */
static const int T83_LEN[14] = { 1920, 1920, 2048, 1536, 1536, 960, 960,
                                 1024, 768, 768, 512, 384, 384, 2048 };
static const int T83_UP[14] = { 1001, 25, 15, 1001, 25, 1001, 25, 15,
                                1001, 25, 15, 1001, 25, 1 };
static const int T83_DOWN[14] = { 960, 24, 16, 960, 24, 960, 24, 16, 960,
                                  24, 16, 960, 24, 1 };

int ac4_frame_len_base(int fs_index, int frame_rate_index)
{
    if (fs_index == 1 && frame_rate_index >= 0 && frame_rate_index < 14)
        return T83_LEN[frame_rate_index];
    if (fs_index == 0 && frame_rate_index == 13)
        return 2048;
    return 0;
}

void ac4_rate_ratio(int fs_index, int frame_rate_index, int *up, int *down)
{
    *up = *down = 1;
    if (fs_index == 1 && frame_rate_index >= 0 && frame_rate_index < 14) {
        *up = T83_UP[frame_rate_index];
        *down = T83_DOWN[frame_rate_index];
    }
}
