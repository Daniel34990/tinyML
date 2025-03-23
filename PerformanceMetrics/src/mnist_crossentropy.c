#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genann.h"
#include "config.h"
#include "mnist_db.h"
#include "matrix.h"
#include "metrics.h"

/*
	Compare 2 modèles.
	Renvoie :
		- 1 si le nouveau modèle est meilleur
		- 0 sinon
*/


int main(int argc, char* argv[]) 
{
    size_t i;
	int j;
	double output[CLASS_COUNT];
    dataset training, tests;

	if(argc != 4) {
		printf("./mnist [NUMBER OF HIDDEN LAYERS] [NEURON PER HIDDEN LAYERS] [OUTPUT FILE] ");
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

    genann *ann = genann_init(training.width * training.height,
		atoi(argv[1]),
		atoi(argv[2]),
		CLASS_COUNT
	);
	assert(ann != NULL);

    genann *old_ann = NULL;

	memset(output, 0, CLASS_COUNT * sizeof(double));

	// Implémentation du "early stopping"
	for(j = 0; is_new_ai_better(old_ann, ann, tests); j ++) {
		// Remplacement de l'ancien modèle
		if(old_ann)
			genann_free(old_ann);
		old_ann = genann_copy(ann);

		// On entraine l'IA sur toute la base de données
		// Ce qui n'est pas vraiment efficace
		for (i = 0; i < training.nimages; ++i) {
			printf("[Entrainement numero %d]: %zd%%\r",
				j+1,
				(100 * (i+1)) / training.nimages
            );
			
			output[training.images[i].class] = 1;
			genann_train(ann, training.images[i].pixels, output, 0.3);
			output[training.images[i].class] = 0;
		}
		printf("\n");
	}

	// Si nous sommes ici, c'est que l'ancien modèle
	// performe mieux que le nouveau
	FILE *output_file = fopen(argv[3], "w");
	if(output_file) {
		genann_write(old_ann, output_file);
		fclose(output_file);
	} else
		perror("fopen");

	genann_free(old_ann);
	genann_free(ann);

	dataset_free(&training);
	dataset_free(&tests);

	return 0;
}

