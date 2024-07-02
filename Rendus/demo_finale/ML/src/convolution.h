#ifndef _CONVOLUTION_H_
#define _CONVOLUTION_H_

#include <stddef.h>

#include "genann.h"
#include "tensor.h"

typedef struct ConvolutionLayer ConvolutionLayer;
struct ConvolutionLayer {
	size_t input_width;
	size_t input_height;
	size_t input_depth;
	
	size_t stride;
	size_t filters;
	size_t pad_width;
	size_t pad_height;

	Tensor *conv_tensor;
	Tensor output_tensor;
	
	Tensor backpropagation_conv_tensor;
	Tensor backpropagation_output_tensor;
};

/* Initialise la couche de convolution */
ConvolutionLayer *convolution_init(
	size_t input_width, size_t input_height, size_t input_depth,
	size_t conv_width, size_t conv_height, size_t filters,
	size_t stride,
	size_t pad_width, size_t pad_height
);

/* Libère la couche de convolution. */
void convolution_free(ConvolutionLayer *);

/* Applique la couche de convolution sur le tenseur inputs */
Tensor convolution_run(ConvolutionLayer *layer, Tensor inputs);

/* Entraine la couche de convolution.
   input correspond à l'entrée donné à la couche de convolution
   lors de l'éxecution, et input_bt_tensor correspond au gradient
   des couches suivantes. */
Tensor convolution_train(ConvolutionLayer *layer,
	Tensor input,
	Tensor input_bt_tensor,
	double learning_rate
);

#endif /* _CONVOLUTION_H_ */
