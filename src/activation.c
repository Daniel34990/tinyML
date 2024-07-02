#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "activation.h"
#include "config.h"

ActivationLayer* activation_init(char* actfun_name, actfun activation, derivate_actfun derivate_act,
	size_t input_width, size_t input_height, size_t input_depth)
{
	ActivationLayer* output;

	assert(actfun_name);
	assert(strlen(actfun_name) < MAX_LENGTH_ACTIVATION_FUNCTION_NAME);

	output = malloc(sizeof(ActivationLayer));
	if(!output) {
		perror("malloc");
		return NULL;
	}

	output->activation = activation;
	output->derivate_act = derivate_act;
	output->actfun_name = actfun_name;
	output->last_input_tainted = 1;

	if(tensor_new(&output->last_input, input_width, input_height, input_depth))
		return NULL;

	if(tensor_new(&output->last_output, input_width, input_height, input_depth))
		return NULL;

	return output;
}

void activation_free(ActivationLayer* al)
{
	tensor_free(al->last_output);
	tensor_free(al->last_input);
	free(al);
}

Tensor activation_run(ActivationLayer *layer, Tensor inputs)
{
	size_t k;

	assert(layer);

	layer->last_input_tainted = 0;
	tensor_copy(&layer->last_input, inputs);

	for(k = 0; k < tensor_size(layer->last_output); k ++)
		layer->last_output.data[k] = layer->activation((void*) layer, inputs.data[k]);
	
	return layer->last_output;
}

Tensor activation_backpropagate(ActivationLayer *layer, Tensor backprop)
{
	size_t k;

	assert(layer);

	if(layer->last_input_tainted == 1)
		return layer->last_input;

	for(k = 0; k < tensor_size(layer->last_input); k ++) {
		layer->last_input.data[k] = layer->derivate_act(layer,
			layer->last_input.data[k],
			layer->last_output.data[k]);
	}
	
	for(k = 0; k < tensor_size(layer->last_input); k ++)
		layer->last_input.data[k] *= backprop.data[k];

	return layer->last_input;
}
