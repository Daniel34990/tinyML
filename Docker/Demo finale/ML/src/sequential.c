#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "sequential.h"
#include "convolution.h"
#include "config.h"

Sequential* sequential_create(size_t input_width, size_t input_height, size_t layer_count)
{
	Sequential* output;

	output = malloc(sizeof(Sequential));
	if(!output) {
		perror("malloc");
		return NULL;
	}

	output->layer_count = layer_count;
	output->input_width = input_width;
	output->input_height = input_height;

	output->layers = malloc(layer_count * sizeof(SequentialLayer));
	if(!output->layers) {
		perror("malloc");
		free(output);
		return NULL;
	}
	bzero(output->layers, layer_count * sizeof(SequentialLayer));

	return output;
}

void sequential_free(Sequential* seq)
{
	size_t i;

	for(i = 0; i < seq->layer_count; i ++) {
		switch(seq->layers[i].type) {
		case SEQUENTIAL_LAYER_NONE: break;
		case SEQUENTIAL_LAYER_FLATTEN:
			free(seq->layers[i].elt);
			break;
		case SEQUENTIAL_LAYER_DENSE:
			genann_free(seq->layers[i].elt);
			break;
		case SEQUENTIAL_LAYER_ACTIVATION_2D:
			activation_free(seq->layers[i].elt);
			break;
		case SEQUENTIAL_LAYER_CONV_2D:
			convolution_free(seq->layers[i].elt);
			break;
		case SEQUENTIAL_LAYER_POOLING_2D:
			pooling_free(seq->layers[i].elt);
			break;
		}
	}

	free(seq->layers);
	free(seq);
}

void sequential_set_pooling_2d(Sequential *seq, size_t index, PoolingLayer *pool)
{
	assert(seq);
	/* La dernière couche sera réservé au réseau de neurone dense */
	assert(index < seq->layer_count - 1);

	seq->layers[index] = (SequentialLayer) {
		.type = SEQUENTIAL_LAYER_POOLING_2D,
		.elt = (void*) pool
	};
}

void sequential_set_dense(Sequential *seq, size_t index, genann const *ann)
{
	assert(seq);
	/* La dernière couche sera réservé au réseau de neurone dense */
	assert(index == seq->layer_count - 1);

	seq->layers[index] = (SequentialLayer) {
		.type = SEQUENTIAL_LAYER_DENSE,
		.elt = (void*) ann
	};
}

void sequential_set_convolution(Sequential *seq, size_t index, ConvolutionLayer const *cl)
{
	assert(seq);
	/* La dernière couche sera réservé au réseau de neurone dense */
	assert(index < seq->layer_count - 1);

	if(index == 0) {
		assert(seq->input_width == cl->input_width);
		assert(seq->input_height == cl->input_height);
	} else {
	/* TODO: Vérifier si les dimensions entre 2 couches sont valides */
	}

	seq->layers[index] = (SequentialLayer) {
		.type = SEQUENTIAL_LAYER_CONV_2D,
		.elt = (void*) cl
	};
}

void sequential_set_activation(Sequential *seq, size_t index, ActivationLayer const *act)
{
	assert(seq);
	/* La dernière couche sera réservé au réseau de neurone dense */
	assert(index < seq->layer_count - 1);

	if(index == 0)
		assert(seq->input_width * seq->input_height == tensor_size(act->last_input));
	else {
	/* TODO: Vérifier si les dimensions entre 2 couches sont valides */
	}

	seq->layers[index] = (SequentialLayer) {
		.type = SEQUENTIAL_LAYER_ACTIVATION_2D,
		.elt = (void*) act
	};
}

void sequential_set_flatten(Sequential *seq, size_t index, FlatteningLayer const *fl)
{
	assert(seq);
	/* La dernière couche sera réservé au réseau de neurone dense */
	assert(index < seq->layer_count - 1);

	seq->layers[index] = (SequentialLayer) {
		.type = SEQUENTIAL_LAYER_FLATTEN,
		.elt = (void*) fl
	};
}

Tensor sequential_run(Sequential const *seq, Tensor input)
{
	size_t i;
	Tensor output;
	
	assert(seq);

	output = input;

	for(i = 0; i < seq->layer_count; i ++) {
		switch(seq->layers[i].type) {
		case SEQUENTIAL_LAYER_DENSE:
			output.data = (double*) genann_run(seq->layers[i].elt, output.data);
			output.width = ((genann*)seq->layers[i].elt)->outputs;
			output.height = 1;
			output.depth = 1;
			break;
			
		case SEQUENTIAL_LAYER_POOLING_2D:
			output = pooling_run(seq->layers[i].elt, output);
			break;
			
		case SEQUENTIAL_LAYER_CONV_2D:
			output = convolution_run(seq->layers[i].elt, output);
			break;
			
		case SEQUENTIAL_LAYER_ACTIVATION_2D:
			output = activation_run(seq->layers[i].elt, output);
			break;

		case SEQUENTIAL_LAYER_FLATTEN:
			output = flatten_run(seq->layers[i].elt, output);
			break;

		case SEQUENTIAL_LAYER_NONE:
			assert(0);
		}
	}

	return output;
}

double const* sequential_retrieve_ann_input(Sequential const *seq, Tensor input)
{
	ConvolutionLayer *cl;
	ActivationLayer *al;
	PoolingLayer *pl;
	genann *ann;
	int k;

	for(k = (int)(seq->layer_count)-2; k >= 0; k --) {
		switch(seq->layers[k].type) {
		case SEQUENTIAL_LAYER_ACTIVATION_2D:
			al = (ActivationLayer*) seq->layers[k].elt;
			return al->last_output.data;

		case SEQUENTIAL_LAYER_POOLING_2D:
			pl = (PoolingLayer*) seq->layers[k].elt;
			return pl->output_tensor.data;
		
		case SEQUENTIAL_LAYER_CONV_2D:
			cl = (ConvolutionLayer*) seq->layers[k].elt;
			return cl->output_tensor.data;

		case SEQUENTIAL_LAYER_FLATTEN:
			break;

		case SEQUENTIAL_LAYER_DENSE:
			ann = (genann*) seq->layers[k].elt;
			return ann->output + ann->hidden * ann->hidden_layers + ann->inputs;

		case SEQUENTIAL_LAYER_NONE:
			assert(0);
		}
	}

	return input.data;
}

void sequential_train(Sequential const *seq, Tensor input, Tensor desired_outputs, double learning_rate)
{
	ActivationLayer *al;
	ConvolutionLayer *cl;
	PoolingLayer *pl;

	Tensor delta;
	Tensor input_tensor;

	sequential_run(seq, input);

	assert(seq->layers[seq->layer_count-1].type == SEQUENTIAL_LAYER_DENSE);

	genann const *last_layer_ann = (genann const *)seq->layers[seq->layer_count-1].elt;
	
	// Le +1 viens du fait que le premier delta est réservé pour le biais
	delta.data = last_layer_ann->delta + 1; // TODO: Vérifier
	delta.width = last_layer_ann->inputs;
	delta.height = 1;
	delta.depth = 1;

	genann_train(
		last_layer_ann,
		sequential_retrieve_ann_input(seq, input),
		desired_outputs.data,
		learning_rate
	);

	for (int i = seq->layer_count - 2; i >= 0; i--) {
        switch(seq->layers[i].type) {	
		case SEQUENTIAL_LAYER_POOLING_2D:
			delta = pooling_backpropagate(
				(PoolingLayer*) seq->layers[i].elt,
				delta
			);
			break;
			
		case SEQUENTIAL_LAYER_ACTIVATION_2D:
			delta = activation_backpropagate(
				(ActivationLayer*) seq->layers[i].elt,
				delta
			);
			break;

		case SEQUENTIAL_LAYER_FLATTEN:
			delta = flatten_backpropagate(
				(FlatteningLayer*) seq->layers[i].elt,
				delta
			);
			break;

		case SEQUENTIAL_LAYER_CONV_2D:
			if (i == 0) {
				cl = (ConvolutionLayer*) seq->layers[i].elt;
				input_tensor = input;
			} else {
				/* TODO: Faire mieux */
				switch(seq->layers[i-1].type) {
				case SEQUENTIAL_LAYER_POOLING_2D:
					pl = (PoolingLayer *) seq->layers[i-1].elt;
					input_tensor = pl->output_tensor;
					break;

				case SEQUENTIAL_LAYER_ACTIVATION_2D:
					al = (ActivationLayer*) seq->layers[i-1].elt;
					cl = (ConvolutionLayer*) seq->layers[i].elt;
					input_tensor = al->last_output;
					break;

				case SEQUENTIAL_LAYER_CONV_2D:
					cl = (ConvolutionLayer*) seq->layers[i-1].elt;
					input_tensor = cl->output_tensor;
					break;

				case SEQUENTIAL_LAYER_FLATTEN:
				case SEQUENTIAL_LAYER_DENSE:
				case SEQUENTIAL_LAYER_NONE:
					assert(0);
				}
			}

			delta = convolution_train(
				(ConvolutionLayer*) seq->layers[i].elt,
				input_tensor, delta,
				learning_rate
			);
			break;

		case SEQUENTIAL_LAYER_NONE:
		case SEQUENTIAL_LAYER_DENSE:
			assert(0);
		}
    }
}

void sequential_write(Sequential const *seq, FILE *out)
{
	size_t i, j;
	PoolingLayer *pl;
	FlatteningLayer *fl;
	ConvolutionLayer *cl;
	ActivationLayer *al;

	fprintf(out, "%zd %zd %zd\n",
		seq->input_width, seq->input_height, seq->layer_count);
	
	/* Je sais que ce n'est pas génial, mais ça fera l'affaire
	   dans le cadre du projet */
	for(i = 0; i < seq->layer_count; i ++) {
		fprintf(out, "%d\n", (int) seq->layers[i].type);
		switch(seq->layers[i].type) {
		case SEQUENTIAL_LAYER_NONE: break;
		case SEQUENTIAL_LAYER_DENSE:
			genann_write((genann*) seq->layers[i].elt, out);
			fputc('\n', out);
			break;
		case SEQUENTIAL_LAYER_ACTIVATION_2D:
			al = (ActivationLayer*) seq->layers[i].elt;
			fprintf(out, "%zu %zu %zu %s\n",
				al->last_input.width,
				al->last_input.height,
				al->last_input.depth,
				al->actfun_name);
			break;
		case SEQUENTIAL_LAYER_POOLING_2D:
			pl = (PoolingLayer*) seq->layers[i].elt;
			fprintf(out, "%d %zu %zu %zu\n",
				(int) pl->type, pl->pool_size, pl->input_size, pl->depth);
			break;
		case SEQUENTIAL_LAYER_FLATTEN:
			fl = (FlatteningLayer*) seq->layers[i].elt;
			fprintf(out, "%zu %zu %zu\n",
				fl->input_width, fl->input_height, fl->input_depth);
			break;
		case SEQUENTIAL_LAYER_CONV_2D:
			cl = (ConvolutionLayer*) seq->layers[i].elt;
			fprintf(out, "%zu %zu %zu %zu %zu %zu %zu %zu %zu\n",
				cl->input_width, cl->input_height, cl->input_depth,
				cl->conv_tensor[0].width, cl->conv_tensor[0].height,
				cl->pad_width, cl->pad_height, cl->filters,
				cl->stride
			);

			for(j = 0; j < cl->filters; j ++)
				tensor_write(cl->conv_tensor[j], out);
			break;
		}
	}
}

Sequential* sequential_read(FILE *in, Sequential_Actfun(*actfun_finder)(const char*))
{
	Sequential *out;
	int tmp;
	size_t i, j;
	size_t in_w, in_h, in_d, layer_count;
	SequentialLayerType layer_type;
	char name_tmp_buf[MAX_LENGTH_ACTIVATION_FUNCTION_NAME+1];
	char *name_buf;
	size_t name_len;

	size_t pool_size, input_size;
	
	size_t pad_w, pad_h;
	size_t conv_w, conv_h;
	size_t filters, stride;
	ConvolutionLayer *cl;

	Sequential_Actfun activation_fun;

	if(fscanf(in, "%zu %zu %zu\n", &in_w, &in_h, &layer_count) != 3)
		return NULL;

	out = sequential_create(in_w, in_h, layer_count);
	for(i = 0; i < layer_count; i ++) {
		fscanf(in, "%d\n", &tmp);
		layer_type = (SequentialLayerType) tmp;
		switch(layer_type) {
		case SEQUENTIAL_LAYER_NONE: break;
		case SEQUENTIAL_LAYER_DENSE:
			sequential_set_dense(out, i, genann_read(in));
			break;
		case SEQUENTIAL_LAYER_FLATTEN:
			fscanf(in, "%zu %zu %zu\n", &in_w, &in_h, &in_d);
			sequential_set_flatten(out, i, flatten_init(in_w, in_h, in_d));
			break;
		case SEQUENTIAL_LAYER_POOLING_2D:
			fscanf(in, "%d %zu %zu %zu\n", &tmp, &pool_size, &input_size, &in_d);
			sequential_set_pooling_2d(out, i, pooling_init((PoolingType) tmp, pool_size, input_size, in_d));
			break;
		case SEQUENTIAL_LAYER_CONV_2D:
			fscanf(in, "%zu %zu %zu %zu %zu %zu %zu %zu %zu\n",
				&in_w, &in_h, &in_d,
				&conv_w, &conv_h,
				&pad_w, &pad_h, &filters,
				&stride);

			cl = convolution_init(in_w, in_h, in_d, conv_w, conv_h, filters, stride, pad_w, pad_h);
			
			for(j = 0; j < filters; j ++)
				tensor_read(cl->conv_tensor[j], in);

			sequential_set_convolution(out, i, cl);
			break;
		case SEQUENTIAL_LAYER_ACTIVATION_2D:
			fscanf(in, "%zu %zu %zu %s\n", &in_w, &in_h, &in_d, (char*) name_tmp_buf);

			// On copie le nom de la fonction dans un buffer définitif
			name_len = strlen(name_tmp_buf);
			name_buf = malloc(name_len + 1);
			memcpy(name_buf, name_tmp_buf, name_len);
			name_buf[name_len] = 0;

			activation_fun = actfun_finder(name_buf);

			sequential_set_activation(out, i,
				activation_init(name_buf, activation_fun.fun,
					activation_fun.derivate,
					in_w, in_h, in_d
				)
			);
			break;
		}
	}

	return out;
}
