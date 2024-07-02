#include "convolution.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

ConvolutionLayer *convolution_init(
	size_t input_width, size_t input_height, size_t input_depth,
	size_t conv_width, size_t conv_height,
	size_t filters, size_t stride,
	size_t pad_width, size_t pad_height)
{
	size_t i;
	ConvolutionLayer *output;

	if((input_width - conv_width) % stride != 0)
		return NULL;
	if((input_height - conv_height) % stride != 0)
		return NULL;

	output = malloc(sizeof(ConvolutionLayer));
	if(!output)
		return NULL;

	output->stride = stride;
	output->filters = filters;
	output->input_width = input_width;
	output->input_height = input_height;
	output->input_depth = input_depth;
	output->pad_width = pad_width;
	output->pad_height = pad_height;

	output->conv_tensor = malloc(sizeof(Tensor) * filters);
	for(i = 0; i < filters; i ++) {
		if(tensor_new(&output->conv_tensor[i], conv_width, conv_height, input_depth))
			return NULL;
		tensor_randomize(&output->conv_tensor[i]);
	}

	if(tensor_new(&output->backpropagation_conv_tensor, conv_width, conv_height, input_depth))
		return NULL;

	if(tensor_new(&output->output_tensor,
		(input_width - conv_width + 2*pad_width) / stride + 1,
		(input_height - conv_height + 2*pad_height) / stride + 1,
		filters
	))
		return NULL;

	if(tensor_new(&output->backpropagation_output_tensor,
		input_width, input_height, input_depth))
		return NULL;

	return output;
}

void convolution_free(ConvolutionLayer *layer)
{
	size_t i;

	if(layer == NULL)
		return;

	for(i = 0; i < layer->filters; i ++)
		tensor_free(layer->conv_tensor[i]);
	free(layer->conv_tensor);

	tensor_free(layer->output_tensor);
	tensor_free(layer->backpropagation_conv_tensor);
	tensor_free(layer->backpropagation_output_tensor);
	free(layer);
}

Tensor convolution_run(ConvolutionLayer *layer, Tensor inputs)
{
	size_t i;

	assert(layer);
	assert(layer->input_width == inputs.width);
	assert(layer->input_height == inputs.height);
	assert(layer->input_depth == inputs.depth);
	
	tensor_clear(layer->output_tensor);

	for(i = 0; i < layer->filters; i ++) {
		tensor_convolution_add(tensor_get_matrix(layer->output_tensor, i),
			layer->conv_tensor[i],
			layer->stride,
			inputs,
			layer->pad_width, layer->pad_height
		);
	}

	return layer->output_tensor;
}

/* in_bt: L'entrée de la backpropagation (L dans la littérature)
   input: L'entrée de la forward propagation */
Tensor convolution_train(ConvolutionLayer *layer, Tensor input, Tensor input_bt_tensor, double learning_rate)
{
	size_t i, j;

	if(!layer)
		return (Tensor) { 0 };
	
	/* Calcul des gradients qui vont être utilisés pour la suite */
	for(i = 0; i < layer->filters; i ++) 
		tensor_rotate_180(layer->conv_tensor[i]);

	tensor_clear(layer->backpropagation_output_tensor);

	for(j = 0; j < layer->filters; j ++) {
		for(i = 0; i < input.depth; i ++) {
			tensor_convolution_add(
				tensor_get_matrix(layer->backpropagation_output_tensor, i),
				tensor_get_matrix(layer->conv_tensor[j], i), layer->stride,
				tensor_get_matrix(input_bt_tensor, j),
				layer->conv_tensor[j].width - 1 - layer->pad_width,
				layer->conv_tensor[j].height - 1 - layer->pad_height
			);
		}
	}

	for(i = 0; i < layer->filters; i ++) 
		tensor_rotate_180(layer->conv_tensor[i]);

	/* Application du gradient descent sur le filtre de convolution */
	for(i = 0; i < layer->filters; i ++) {
		tensor_clear(layer->backpropagation_conv_tensor);
		for(j = 0; j < layer->input_depth; j ++) {
			tensor_convolution_add(tensor_get_matrix(layer->backpropagation_conv_tensor, j),
				tensor_get_matrix(input_bt_tensor, i), layer->stride,
				tensor_get_matrix(input, j),
				layer->pad_width, layer->pad_height);
		}

		tensor_scale_and_add(layer->conv_tensor[i], layer->backpropagation_conv_tensor, -learning_rate);
	}

	return layer->backpropagation_output_tensor;
}
