#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "genann.h"
#include "config.h"
#include "mnist_db.h"
#include "tensor.h"
#include "pooling.h"

PoolingLayer* pooling_init(PoolingType type, size_t pool_size, size_t input_size, size_t depth)
{
    PoolingLayer* P = malloc(sizeof(PoolingLayer));
    if (P == NULL) {
        perror("malloc");
        return NULL; // Retourne NULL en cas d'échec
    }

	const size_t output_size = input_size / pool_size;

    P->type = type;
    P->depth = depth;
    P->pool_size = pool_size;
    P->input_size = input_size;

	if(tensor_new(&P->output_tensor, output_size, output_size, depth)) {
		free(P);
		return NULL;
	}
	
	if(tensor_new(&P->backpropagation_tensor, input_size, input_size, depth)) {
		tensor_free(P->output_tensor);
		free(P);
		return NULL;
	}

	if(type == MAX_POOLING) {
		P->selected_indices = malloc(output_size * output_size * depth * sizeof(int));
		if(!P->selected_indices) {
        	perror("malloc");
			tensor_free(P->backpropagation_tensor);
			tensor_free(P->output_tensor);
			free(P);
			return NULL;
		}
	}

    return P;
}

void pooling_free(PoolingLayer* p)
{
	tensor_free(p->backpropagation_tensor);
	tensor_free(p->output_tensor);
	free(p);
}

Tensor pooling_run(PoolingLayer *pool, Tensor inputs)
{
    size_t i, j, k, x, y;
	size_t x0, y0, z0;
	size_t max_index;
    double max,sum;
    
    assert(pool->input_size % pool->pool_size == 0);
    
    const size_t input_size = pool->input_size;
    const size_t output_size = pool->input_size / pool->pool_size;
    double *out = pool->output_tensor.data;
    
	for(k = 0; k < pool->depth; k ++) {
		z0 = k * input_size * input_size;

		for(i = 0; i < output_size; i++) {
			x0 = i * pool->pool_size;

			for(j = 0; j < output_size; j++) {
				y0 = j * pool->pool_size;

				switch(pool->type) {
				case MAX_POOLING:
					max = inputs.data[z0 + y0 * input_size + x0]; // Initialise max avec la première valeur du bloc
					max_index = 0;
					for(y = 0; y < pool->pool_size; y++) {
						for(x = 0; x < pool->pool_size; x++) {
							double val = inputs.data[z0 + (y0 + y) * input_size + (x0 + x)];
							if (val > max) {
								max = val;
								max_index = y * pool->pool_size + x;
							}
						}
					}
					out[(k * output_size + i) * output_size + j] = max;
					pool->selected_indices[(k * output_size + i) * output_size + j] = max_index;
					break;

				case AVERAGE_POOLING:
					sum = 0.0;
					for(y = 0; y < pool->pool_size; y++) {
						for(x = 0; x < pool->pool_size; x++) {
							sum += inputs.data[z0 + (y0 + y) * input_size + (x0 + x)];
						}
					}
					out[(k * output_size + i) * output_size + j] = sum / (pool->pool_size * pool->pool_size);
					break;
				}
			}
		}
	}
    
    return pool->output_tensor;
}    

Tensor pooling_backpropagate(PoolingLayer *pool, Tensor backprop)
{
	size_t i, j, k;
	size_t x, y;
	size_t x0, y0, z0;
	size_t index;
	double coeff;
    const size_t output_size = (int) pool->input_size / pool->pool_size;
    double *out = pool->backpropagation_tensor.data;

	if(pool->type != AVERAGE_POOLING)
		tensor_clear(pool->backpropagation_tensor);

    for(k = 0; k < pool->depth; k++) {
		z0 = k * pool->input_size * pool->input_size;

		for(i = 0; i < output_size; i++) {
			x0 = i * pool->pool_size;

			for(j = 0; j < output_size; j++) {
				y0 = j * pool->pool_size;

				switch(pool->type) {
				case MAX_POOLING:
					index = pool->selected_indices[i * output_size + j];
					x = index % pool->pool_size;
					y = index / pool->pool_size;
					out[z0 + (y0 + y) * pool->input_size + (x0 + x)] = backprop.data[j * output_size + i];
					break;

				case AVERAGE_POOLING:
					coeff = backprop.data[j * output_size + i] / ((double)(pool->pool_size * pool->pool_size));
					for(y = 0; y < pool->pool_size; y++) {
						for(x = 0; x < pool->pool_size; x++)
							out[z0 + (y0 + y) * pool->input_size + (x0 + x)] = coeff;
					}
					break;
				}
			}
		}
	}

	return pool->backpropagation_tensor;
}
