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

    FILE *fp = fopen("confusion.csv", "w+");
    if (fp == NULL) {
    perror("Erreur lors de l'ouverture du fichier");
    exit(1);
    }
    // En-tête du fichier CSV
    fprintf(fp, "id,vrai_label,pred_label\n");

    /*for (int hidden_layers=1;hidden_layers<20;hidden_layers++){
        for (int hidden=1;hidden<20;hidden++){
            printf("%d,%d\n",hidden_layers,hidden);
            fit_model(training,tests,hidden_layers,hidden,fp);
        }
    }*/
	conf_data(training,tests,2,9,fp);
    
    

	dataset_free(&training);
	dataset_free(&tests);
    fclose(fp);
	return 0;
}