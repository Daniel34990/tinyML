#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "genann.h"
#include "config.h"
#include "mnist_db.h"
#include "matrix.h"
#include "mesures.h"
#include <time.h>
#include <sys/resource.h>

dataset reduced_dataset(dataset original_dataset, int nimages)
{
    dataset reduced_dataset;
    reduced_dataset.nimages = 100; // Nombre d'images dans le nouveau dataset réduit.
    reduced_dataset.width = original_dataset.width; // Copie de la largeur.
    reduced_dataset.height = original_dataset.height; // Copie de la hauteur.
    reduced_dataset.images = (image *)malloc(sizeof(image) * reduced_dataset.nimages); // Allocation de l'espace pour n images.
    for (size_t i = 0; i < reduced_dataset.nimages; ++i) {
        reduced_dataset.images[i].pixels = malloc(sizeof(double) * original_dataset.width * original_dataset.height);
        memcpy(reduced_dataset.images[i].pixels, original_dataset.images[i].pixels, sizeof(double) * original_dataset.width * original_dataset.height);
        reduced_dataset.images[i].class = original_dataset.images[i].class;
}
    return reduced_dataset;
}


double categorical_crossentropy(genann* net,dataset tests)
{
    size_t i,j;
    size_t max_index;
    int correct_predictions=0;
    double validation_loss=0.0;
    const double* out;
    for (i=0; i<tests.nimages; i++) {
        out=genann_run(net,tests.images[i].pixels);
        max_index=0;
        for (j=0; j<CLASS_COUNT; j++) {
            if (out[j]>out[max_index])
                max_index=j;
        }
        validation_loss-=log(out[max_index]);
        if (tests.images[i].class == max_index)
            correct_predictions++;
        
    }
    return validation_loss;
}

/*
	Compare 2 modèles.
	Renvoie :
		- 1 si le nouveau modèle est meilleur
		- 0 sinon
*/

double accuracy(genann* ann, dataset tests){
    size_t i, j;
	size_t max_index;
	double trace_matrix;
	const double *out;
	Matrix confusion_matrix;

	matrix_new(&confusion_matrix, CLASS_COUNT, CLASS_COUNT);
	// Un modèle, tant qu'il est initialisé
	// sera toujours mieux que rien.
	
    for (i = 0; i < tests.nimages; i ++) {
		out = genann_run(ann, tests.images[i].pixels);
		

		// On utilise argmax
		// Car c'est plus simple et moins couteux à implémenter
		max_index = 0;
		for(j = 1; j < CLASS_COUNT; j ++) {
			if(out[j] > out[max_index]) 
				max_index = j;
		}

		confusion_matrix.data[CLASS_COUNT * max_index + tests.images[i].class] ++;
		

		printf("Tests: %zd%%\r", (100 * (i+1)) / tests.nimages);
	}
	printf("\n");

	// TODO: Utiliser une meilleur mesure
	// que la trace de la matrice de confusion
	trace_matrix = matrix_trace(confusion_matrix);

    return trace_matrix/tests.nimages;
}

int is_new_ai_better(genann *old, genann *new, dataset tests)
{
    assert(new);

	// Un modèle, tant qu'il est initialisé
	// sera toujours mieux que rien.
	if(old == NULL)
		return 1;
	
	double validation_loss_new, validation_loss_old;
	const double *out_new, *out_old;
	validation_loss_new=categorical_crossentropy(new,tests);
	validation_loss_old=categorical_crossentropy(old,tests);
	printf("categorical_crossentropy_new: %lf\n",validation_loss_new);
	printf("categorical_crossentropy_old: %lf\n",validation_loss_old);

	if(validation_loss_old > validation_loss_new)
		return 1;
	else
		return 0;
}

void fit_model(dataset training, dataset tests, int hidden_layers, int hidden, FILE*fp)
{
    size_t i;
	int j;
	double output[CLASS_COUNT];

    struct timespec start, end;
    struct rusage usage_start, usage_end;
    
    

    dataset tests_reduced=reduced_dataset(tests, 25);

    clock_gettime(CLOCK_MONOTONIC, &start);
    getrusage(RUSAGE_SELF, &usage_start);

    genann *ann = genann_init(training.width * training.height,
		hidden_layers,
		hidden,
		CLASS_COUNT
	);
	assert(ann != NULL);

    genann *old_ann = NULL;

	memset(output, 0, CLASS_COUNT * sizeof(double));

	// Implémentation du "early stopping"
	for(j = 0; is_new_ai_better(old_ann, ann, tests) ; j ++) {
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
			genann_train(ann, training.images[i].pixels, output, 0.1);
            output[training.images[i].class] = 0;
            
		}
        
        
		printf("\n");
} 
    clock_gettime(CLOCK_MONOTONIC, &end);
    getrusage(RUSAGE_SELF, &usage_end);
   
    double training_time = end.tv_sec - start.tv_sec;
    training_time += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    float memory_used = (usage_end.ru_maxrss)/1024;

    int total_weights = (784 + 1) * hidden // Poids de l'entrée vers la première couche cachée
    + (hidden + 1) * hidden * (hidden_layers - 1) // Poids entre les couches cachées
    + (hidden + 1) * 10; // Poids de la dernière couche cachée vers la sortie
    int total_neurons = 784 + hidden * hidden_layers + 10;
    double total_memory = (sizeof(genann) + (total_weights + total_neurons * 2) * sizeof(double))/1024;

    clock_gettime(CLOCK_MONOTONIC, &start);
    for (i=0;i<tests.nimages;i++){
        genann_run(old_ann,training.images[i].pixels);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    double running_time = end.tv_sec - start.tv_sec;
    running_time += (end.tv_nsec - start.tv_nsec) / 1000000000.0;
    running_time*=1000;

    double acc = accuracy(old_ann,tests)*100;
    double ratio=acc/training_time;

    fprintf(fp, "%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f\n", 784, hidden_layers,
    hidden, 10, memory_used,total_memory,training_time,running_time,acc,ratio);
    
    genann_free(old_ann);
	genann_free(ann);
}
