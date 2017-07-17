#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include "Maxfiles.h"
#include "MaxSLiCInterface.h"


int main(void)
{
	const int CONV_H = 10;
	const int CONV_W = 10;
	const int CONV_K = 3;
	const int POOL_H = 8;
	const int POOL_W = 8;

	const int NUM_BATCHES = 10;
	const int IMAGE_SIZE_IN_BYTES = NUM_BATCHES * CONV_H * CONV_W * sizeof(int32_t);
	const int LOGITS_SIZE_IN_BYTES = NUM_BATCHES * POOL_H * POOL_W / 4 * sizeof(int32_t);
	int32_t *image = malloc(IMAGE_SIZE_IN_BYTES);
	int32_t *logits = malloc(LOGITS_SIZE_IN_BYTES);
	int32_t *expected = malloc(LOGITS_SIZE_IN_BYTES);
	int32_t coeff[CONV_K * CONV_K];

	for(int i = 0; i < NUM_BATCHES * CONV_H * CONV_W; ++i)
		image[i] = random() % 100;
	for (int i = 0; i < CONV_K * CONV_K; i ++)
		coeff[i] = random() % 100;


	// EXPECTED
	int32_t *convOut = malloc(POOL_H * POOL_W * NUM_BATCHES * sizeof(int32_t));

	// CONV
	for (int i = 0; i < NUM_BATCHES; i ++) {
		for (int h = 0; h < POOL_H; h ++) {
			for (int w = 0; w < POOL_W; w ++) {
				convOut[i * POOL_H * POOL_W + h * POOL_W + w] = 0;
				for (int ki = 0; ki < CONV_K; ki ++)
					for (int kj = 0; kj < CONV_K; kj ++) {
						int32_t weight = coeff[ki * CONV_K + kj];
						int ih = h + ki - CONV_K / 2 + 1;
						int iw = w + kj - CONV_K / 2 + 1;
						// printf("%d %d\n", ih, iw);
						int32_t I = image[i * CONV_H * CONV_W + ih * CONV_W + iw];
						convOut[i * POOL_H * POOL_W + h * POOL_W + w] += I * weight;
					}
			}
		}
	}

	// POOL
	for (int i = 0; i < NUM_BATCHES; i ++) {
			for (int h = 0; h < POOL_H; h += 2) {
				for (int w = 0; w < POOL_W; w += 2) {
					int32_t max = INT_MIN;

					for (int ki = 0; ki < 2; ki ++)
						for (int kj = 0; kj < 2; kj ++) {
							int32_t tmp = convOut[i * POOL_H * POOL_W + (h + ki) * POOL_W + (w + kj)];
							max = (max > tmp) ? max : tmp;
						}
					expected[i * POOL_H * POOL_W / 4 + h / 2 * POOL_W / 2 + w / 2] = max;

				}
			}
		}


	printf("Running on DFE.\n");
	MaxCNN(NUM_BATCHES, coeff[0], coeff[1], coeff[2], coeff[3], coeff[4], coeff[5], coeff[6], coeff[7], coeff[8], image, logits);

	printf("Done.\n");
	for (int i = 0; i < NUM_BATCHES * POOL_H * POOL_W / 4; i ++) {
		printf("logits[%3d] = %d expected = %d\n", i, logits[i], expected[i]);
		if (logits[i] != expected[i]) {
			exit(1);
		}
	}

	return 0;
}
