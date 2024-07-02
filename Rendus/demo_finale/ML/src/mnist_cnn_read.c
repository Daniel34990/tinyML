#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <time.h>

#include "config.h"
#include "mnist_db.h"
#include "sequential.h"

#define CLASS_COUNT 10

double sigmoid(void *userdata, double x)
{ return 1.0 / (1 + exp(-x)); }

double sigmoid_derivate(void *userdata, double x, double y)
{ return y * (1. - y); }

void model_test(Sequential *model, MnistDataset tests)
{
    size_t i, j;
	const double *out;
	int continue_test;

	size_t loaded_batch_size;
	size_t max_index;
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

	/*
	 * TODO: Utiliser une meilleur mesure
	 * que la trace de la matrice de confusion
	 */
	trace_matrix = 0;
	for(i = 0; i < CLASS_COUNT; i ++)
		trace_matrix += confusion_matrix[i][i];

	printf("Model: %lf correct predictions\n", trace_matrix);
}

Sequential_Actfun activation_function_finder(const char* name)
{
	Sequential_Actfun output;

	output.fun = NULL;
	output.derivate = NULL;

	if(!strcmp(name, "sigmoid")) {
		output.fun = sigmoid;
		output.derivate = sigmoid_derivate;
	}

	return output;
}

int main(void) 
{
    MnistDataset tests;
	Sequential *model;

	if(mnist_init(&tests,
		"mnist_data/t10k-images-idx3-ubyte",
		"mnist_data/t10k-labels-idx1-ubyte",
		1, 0, 512))
		return 1;
	mnist_add_noise(&tests, 0.1);

	srand(time(NULL));

	FILE *f = fopen("test.txt", "r");
	if(!f) {
		perror("fopen");
		return -1;
	}

	model = sequential_read(f, activation_function_finder);
	fclose(f);

	model_test(model, tests);

	sequential_free(model);
	mnist_free(&tests);

	return 0;
}
