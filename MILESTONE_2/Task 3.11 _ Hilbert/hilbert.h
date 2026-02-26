#ifndef HILBERT_H
#define HILBERT_H

#define C 1
#define H 64
#define W 64
#define SEQ_LEN 4096

void hilbert_scan(float image[C][H][W],
                  int* hilbert_indices,
                  float sequence[SEQ_LEN][C]);

#endif