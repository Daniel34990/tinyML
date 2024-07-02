#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <stdlib.h>
#include <math.h>

#include "config.h"
#include "mnist_db.h"
#include "sequential.h"

#define CLASS_COUNT 10

double sigmoid(void *userdata, double x)
{ return 1.0 / (1 + exp(-x)); }

double sigmoid_derivate(void *userdata, double x, double y)
{ return y * (1. - y); }

int model_init(Sequential **output)
{
	ConvolutionLayer *c1, *c2;
	PoolingLayer *p1, *p2;
	ActivationLayer *a1, *a2;
	FlatteningLayer *fl;
	genann *ann;
	assert(output);

	/* The model is an implementation of LeNet architecture */
	c1 = convolution_init(28, 28, 1, 5, 5, /*filters=*/6, 1, 2, 2);
	a1 = activation_init((char*)"sigmoid", sigmoid, sigmoid_derivate, 28, 28, 6);
	p1 = pooling_init(AVERAGE_POOLING, 2, 28, 6);
	c2 = convolution_init(14, 14, 6, 5, 5, /*filters=*/16, 1, 2, 2);
	a2 = activation_init((char*)"sigmoid", sigmoid, sigmoid_derivate, 14, 14, 16);
	p2 = pooling_init(AVERAGE_POOLING, 2, 14, 16);
	fl = flatten_init(7, 7, 16);
	ann = genann_init(7*7*16, 2, 120, CLASS_COUNT);

	*output = sequential_create(28, 28, 8);
	sequential_set_convolution(*output, 0, c1);
	sequential_set_activation(*output, 1, a1);
	sequential_set_pooling_2d(*output, 2, p1);
	sequential_set_convolution(*output, 3, c2);
	sequential_set_activation(*output, 4, a2);
	sequential_set_pooling_2d(*output, 5, p2);
	sequential_set_flatten(*output, 6, fl);
	sequential_set_dense(*output, 7, ann);

	return 0;
}

void model_test(Sequential *model, MnistDataset tests)
{
    size_t i, j;
	size_t max_index;
	size_t loaded_batch_size;
	int continue_test;
	const double *out;
	Tensor input_tensor;

	double trace_matrix;
	size_t confusion_matrix[CLASS_COUNT][CLASS_COUNT];

	bzero(confusion_matrix, sizeof(confusion_matrix));

	assert(model);

	input_tensor.width = 28;
	input_tensor.height = 28;
	input_tensor.depth = 1;

	continue_test = 1;
	while(continue_test) {
		loaded_batch_size = mnist_load_batch(&tests);
		continue_test = (tests.entries_read < tests.entries_count);
		for (i = 0; i < loaded_batch_size; i ++) {
			input_tensor.data = tests.batch_entries[i].pixels;
			out = sequential_run(model, input_tensor).data;

			/*
			 * On utilise argmax
			 * Car c'est plus simple et moins couteux à implémenter
			 */
			max_index = 0;
			for(j = 1; j < CLASS_COUNT; j ++) {
				if(out[j] > out[max_index]) 
					max_index = j;
			}

			confusion_matrix[max_index][tests.batch_entries[i].class] ++;

			printf("Tests: %zd%%\r", (100 * (tests.entries_read - loaded_batch_size + i)) / tests.entries_count);
		}
	}
	printf("\n");


	// TODO: Utiliser une meilleur mesure
	// que la trace de la matrice de confusion
	trace_matrix = 0;
	for(i = 0; i < CLASS_COUNT; i ++)
		trace_matrix += confusion_matrix[i][i];

	printf("Model: %lf correct predictions\n", trace_matrix);
}

int main(void) 
{
    size_t i;
	size_t training_count;
	size_t batch_size;
	int continue_training;
    MnistDataset training, tests;
	Sequential *model;
	Tensor input_image_tensor;
	Tensor wanted_output;

	if(mnist_init(&training,
		"mnist_data/train-images-idx3-ubyte",
		"mnist_data/train-labels-idx1-ubyte",
		1, 0, 256
	))
		return 1;
    
	if(mnist_init(&tests,
		"mnist_data/t10k-images-idx3-ubyte",
		"mnist_data/t10k-labels-idx1-ubyte",
		1, 0, 256
	)) {
		mnist_free(&training);
		return 1;
	}

	assert(training.width == tests.width);
	assert(training.height == tests.height);
	assert(training.width != 0);
	assert(training.height != 0);

	if(tensor_new(&wanted_output, CLASS_COUNT, 1, 1)) {
		return 1;
	}

	if(model_init(&model)) {
		mnist_free(&training);
		mnist_free(&tests);
		return 1;
	}

	input_image_tensor.width = 28;
	input_image_tensor.height = 28;
	input_image_tensor.depth = 1;
	for(training_count = 1; training_count < 2; training_count ++) {
		continue_training = 1;
		while(continue_training) {
			batch_size = mnist_load_batch(&training);
			continue_training = (training.entries_read < training.entries_count);
			for (i = 0; i < batch_size; ++i) {
				printf("[Entrainement numéro %zu]: %zd%% (%zd / %zd)\r",
					training_count,
					(100 * (training.entries_read - training.batch_size + i + 1)) / training.entries_count,
					training.entries_read - training.batch_size + i + 1,
					training.entries_count
				);
				
				wanted_output.data[training.batch_entries[i].class] = 1;
				input_image_tensor.data = training.batch_entries[i].pixels;
				sequential_train(model, input_image_tensor, wanted_output, 0.25);
				wanted_output.data[training.batch_entries[i].class] = 0;
			}
		}
		printf("\n");
		model_test(model, tests);
	}

	FILE *f = fopen("test.txt", "wb");
	sequential_write(model, f);
	fclose(f);

	sequential_free(model);

	mnist_free(&training);
	mnist_free(&tests);
	
	tensor_free(wanted_output);

	return 0;
}
