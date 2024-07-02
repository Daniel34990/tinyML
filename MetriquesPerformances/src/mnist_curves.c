#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genann.h"
#include "config.h"
#include "mnist_db.h"
#include "matrix.h"
#include "metrics.h"



int main(int argc, char* argv[]) 
{
    
    dataset training, tests;

	if(argc != 3) {
		printf("./mnist [NUMBER OF HIDDEN LAYERS] [NEURON PER HIDDEN LAYERS] ");
		return 1;
	}

	if(dataset_read(&training,
		"mnist_data/train-images-idx3-ubyte",
		"mnist_data/train-labels-idx1-ubyte")
	)
		return 1;
    
	if(dataset_read(&tests,
		"mnist_data/t10k-images-idx3-ubyte",
		"mnist_data/t10k-labels-idx1-ubyte")
	) {
		dataset_free(&training);
		return 1;
	}

	assert(training.width == tests.width);
	assert(training.height == tests.height);
	assert(training.width != 0);
	assert(training.height != 0);

    fit_model(training,tests,atoi(argv[1]),atoi(argv[2]));

	dataset_free(&training);
	dataset_free(&tests);

	return 0;
}