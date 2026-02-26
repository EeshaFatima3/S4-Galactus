#include "hilbert.h"

void hilbert_scan(float image[C][H][W],
                  int* hilbert_indices,
                  float sequence[SEQ_LEN][C])
{
    for (int d = 0; d < SEQ_LEN; d++) {

        int idx = hilbert_indices[d];
        int row = idx / W;
        int col = idx % W;

        for (int c = 0; c < C; c++) {
            sequence[d][c] = image[c][row][col];
        }
    }
}