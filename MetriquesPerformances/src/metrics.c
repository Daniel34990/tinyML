#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "genann.h"
#include "config.h"
#include "mnist_db.h"
#include "matrix.h"
#include "metrics.h"
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

void conf_data(dataset training, dataset tests,int hidden_layers, int hidden,FILE* fp){
    int i;
	int j;
	double output[CLASS_COUNT];
    const double * output_run;
    double crossentropy_loss;
    double validation_loss;
    const double *out;
    int max_index;

    dataset tests_reduced=reduced_dataset(tests, 25);

    genann *ann = genann_init(training.width * training.height,
		hidden_layers,
		hidden,
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
    for (i = 0; i < tests.nimages; i ++) {
		out = genann_run(ann, tests.images[i].pixels);
		

		// On utilise argmax
		// Car c'est plus simple et moins couteux à implémenter
		max_index = 0;
		for(j = 1; j < CLASS_COUNT; j ++) {
			if(out[j] > out[max_index]) 
				max_index = j;
		}

		fprintf(fp,"%d,%d,%d\n",i,tests.images[i].class,max_index);
		

		printf("Tests: %zd%%\r", (100 * (i+1)) / tests.nimages);
	}
	printf("\n");
    fclose(fp);
    genann_free(old_ann);
	genann_free(ann);    
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

void fit_model(dataset training, dataset tests, int hidden_layers, int hidden)
{
    size_t i;
	int j;
	double output[CLASS_COUNT];
    const double * output_run;
    double crossentropy_loss;
    double validation_loss;

    dataset tests_reduced=reduced_dataset(tests, 25);
    
    char filename[100];
    sprintf(filename, "training_curve_%dhiddenlayers_%dhidden.csv", hidden_layers, hidden);
    FILE *fp = fopen(filename, "w+");
    if (fp == NULL) {
    perror("Erreur lors de l'ouverture du fichier");
    exit(1);
    }


    // En-tête du fichier CSV
    fprintf(fp, "generation,cross_entropy,validation_error\n");

    genann *ann = genann_init(training.width * training.height,
		hidden_layers,
		hidden,
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
			genann_train(ann, training.images[i].pixels, output, 0.1);
            output[training.images[i].class] = 0;
            if (i%500==0){
            output_run=genann_run(ann,training.images[i].pixels);
            crossentropy_loss=-log(output_run[training.images[i].class]);
            validation_loss=categorical_crossentropy(ann,tests_reduced);
            fprintf(fp, "%zu,%.3f,%.3f\n", training.nimages*j+i, crossentropy_loss, validation_loss);
            }
            
		}
        
        
		printf("\n");
} 
    fclose(fp);
    genann_free(old_ann);
	genann_free(ann);
}

