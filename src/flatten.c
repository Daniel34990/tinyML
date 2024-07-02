#include <stdlib.h>

#include "flatten.h"


FlatteningLayer *flatten_init(size_t input_width, size_t input_height, size_t input_depth)
{
	FlatteningLayer *output;
	
	output = malloc(sizeof(FlatteningLayer));
	if(!output)
		return NULL;

	output->input_width = input_width;
	output->input_height = input_height;
	output->input_depth = input_depth;

	return output;
}

Tensor flatten_run(FlatteningLayer *fl, Tensor input)
{
	Tensor output;

	output.data = input.data;
	output.width = fl->input_width * fl->input_height * fl->input_depth;
	output.height = 1;
	output.depth = 1;

	return output;
}

Tensor flatten_backpropagate(FlatteningLayer *fl, Tensor bt_input)
{
	Tensor output;

	output.data = bt_input.data;
	output.width = fl->input_width;
	output.height = fl->input_height;
	output.depth = fl->input_depth;

	return output;
}
