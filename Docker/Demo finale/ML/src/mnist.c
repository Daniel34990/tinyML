#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#include "genann.h"
#include "config.h"
#include "mnist_db.h"

#define CLASS_COUNT 10

/*
	Compare 2 modèles.
	Renvoie :
		- 1 si le nouveau modèle est meilleur
		- 0 sinon
*/
int is_new_ai_better(genann *old, genann *new, MnistDataset tests)
{
    size_t i, j;
	size_t max_index_new, max_index_old;
	const double *out_new, *out_old;

	size_t trace_matrix_new, trace_matrix_old;
	size_t old_confusion_matrix[CLASS_COUNT][CLASS_COUNT];
	size_t new_confusion_matrix[CLASS_COUNT][CLASS_COUNT];

	bzero(old_confusion_matrix, sizeof(old_confusion_matrix));
	bzero(new_confusion_matrix, sizeof(new_confusion_matrix));

	assert(new);

	// Un modèle, tant qu'il est initialisé
	// sera toujours mieux que rien.
	if(old == NULL)
		return 1;
	
	for (i = 0; i < tests.batch_size; i ++) {
		out_new = genann_run(new, tests.batch_entries[i].pixels);
		out_old = genann_run(old, tests.batch_entries[i].pixels);

		// On utilise argmax
		// Car c'est plus simple et moins couteux à implémenter
		max_index_new = 0;
		for(j = 1; j < CLASS_COUNT; j ++) {
			if(out_new[j] > out_new[max_index_new]) 
				max_index_new = j;
		}

		max_index_old = 0;
		for(j = 1; j < CLASS_COUNT; j ++) {
			if(out_old[j] > out_old[max_index_old]) 
				max_index_old = j;
		}

		new_confusion_matrix[max_index_new][tests.batch_entries[i].class] ++;
		old_confusion_matrix[max_index_old][tests.batch_entries[i].class] ++;

		printf("Tests: %zd%%\r", (100 * (i+1)) / tests.batch_size);
	}
	printf("\n");

	// TODO: Utiliser une meilleur mesure
	// que la trace de la matrice de confusion
	trace_matrix_new = 0;
	trace_matrix_old = 0;
	for(i = 0; i < CLASS_COUNT; i ++) {
		trace_matrix_new += new_confusion_matrix[i][i];
		trace_matrix_old += old_confusion_matrix[i][i];
	}

#ifdef PRINT_PERFORMANCE
	printf("Old model: %zu correct predictions\n", trace_matrix_old);
	printf("New model: %zu correct predictions\n", trace_matrix_new);
#endif

	if(trace_matrix_old < trace_matrix_new)
		return 1;
	else
		return 0;
}

int main(int argc, char* argv[]) 
{
    size_t i;
	int j;
	double output[CLASS_COUNT];
    MnistDataset training, tests;

	if(argc != 4) {
		printf("./mnist [NUMBER OF HIDDEN LAYERS] [NEURON PER HIDDEN LAYERS] [OUTPUT FILE]");
		return 1;
	}

	if(mnist_init(&training,
		"mnist_data/train-images-idx3-ubyte",
		"mnist_data/train-labels-idx1-ubyte",
		0, 0, 0
	))
		return 1;

	if(mnist_load_batch(&training) != training.batch_size) {
		mnist_free(&training);
		return 1;
	}
    
	if(mnist_init(&tests,
		"mnist_data/t10k-images-idx3-ubyte",
		"mnist_data/t10k-labels-idx1-ubyte",
		0, 0, 0
	)) {
		mnist_free(&training);
		return 1;
	}

	if(mnist_load_batch(&tests) != tests.batch_size) {
		mnist_free(&tests);
		mnist_free(&training);
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
		for (i = 0; i < training.batch_size; ++i) {
			printf("[Entrainement numero %d]: %zd%%\r",
				j+1,
				(100 * (i+1)) / training.batch_size
            );
			
			output[training.batch_entries[i].class] = 1;
			genann_train(ann, training.batch_entries[i].pixels, output, 0.25);
			output[training.batch_entries[i].class] = 0;
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

	mnist_free(&training);
	mnist_free(&tests);

	return 0;
}

