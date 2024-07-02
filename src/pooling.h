#ifndef _POOLING_H_
#define _POOLING_H_

#include <stddef.h>

#include "tensor.h"

enum PoolingType {
	MAX_POOLING,
	AVERAGE_POOLING
};
typedef enum PoolingType PoolingType;

typedef struct PoolingLayer PoolingLayer;
struct PoolingLayer {
	PoolingType type;
	size_t pool_size;

	size_t input_size;
	size_t depth;

	Tensor output_tensor;
	Tensor backpropagation_tensor;
    int *selected_indices;
};

/* Initialise la couche de pooling */
PoolingLayer* pooling_init(PoolingType type, size_t pool_size, size_t input_size, size_t depth);

/* Libère la couche de pooling de la mémoire */
void pooling_free(PoolingLayer*);

/* Applique la couche de pooling sur l'entrée inputs */
Tensor pooling_run(PoolingLayer *pool, Tensor inputs);

/* Calcule le gradient de la couche en fonction du gradient des
   des couches suivants "backprop". */
Tensor pooling_backpropagate(PoolingLayer *pool, Tensor backprop);

#endif /* _POOLING_H_ */
