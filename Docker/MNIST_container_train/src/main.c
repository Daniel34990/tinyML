#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "genann.h"

int renverse(int a)
{
    return
        ((a >>  0) & 0xFF) << 24 |
        ((a >>  8) & 0xFF) << 16 |
        ((a >> 16) & 0xFF) <<  8 |
        ((a >> 24) & 0xFF) <<  0;
}

struct image {
    int class;
    double *pixels;
};
typedef struct image image;

struct dataset {
    size_t nimages;
    int width;
    int height;
    image *images;
};
typedef struct dataset dataset;

const dataset empty_set = (dataset) {
    .nimages = 0,
	.width = 0,
	.height = 0,
    .images = NULL
};

int dataset_read(dataset *output, char *images_file, char *labels_file)
{
	char c;
	size_t i;
    FILE *fimage;

	if(!output)
		return -1;

    *output = empty_set;

    fimage = fopen(images_file, "r");
    if(!fimage) {
        perror("dataset_read : fopen failed");
		return -1;
    }

	fseek(fimage, 4, SEEK_SET);

    if(!fread(&output->nimages, 4, 1, fimage)) {
        perror("fread1");
		return -1;
    }
	
    if(!fread(&output->width, 4, 1, fimage)) {
        perror("fread2");
		return -1;
    }

    if(!fread(&output->height, 4, 1, fimage)) {
        perror("fread3");
		return -1;
    }

    output->nimages = renverse(output->nimages);
    output->width  = renverse(output->width);
    output->height = renverse(output->height);

	printf("Image count: %ld; Width: %d; Height: %d\n",
		output->nimages, output->width, output->height);

    output->images = malloc(sizeof(image) * output->nimages);
    if(!output->images) {
        perror("malloc");
		return -1;
    }

	unsigned char* raw_buffer = malloc(sizeof(unsigned char) * output->width * output->height * output->nimages);
	if(fread(raw_buffer, sizeof(unsigned char) * output->width * output->height, output->nimages, fimage) != output->nimages) {
		perror("fread4");
		return -1;
	}

	double* buffer = malloc(sizeof(double) * output->width * output->height * output->nimages);
	for(i = 0; i < output->width * output->height * output->nimages; i ++)
		buffer[i] = (double)(raw_buffer[i]) / 255.;

	free(raw_buffer);

    for(i = 0; i < output->nimages; i ++) {
		output->images[i].pixels = &buffer[output->width * output->height * i];
    }

    fclose(fimage);

    FILE *flabels = fopen(labels_file, "r");
	fseek(flabels, 8, SEEK_SET);

	for(i = 0; i < output->nimages; i ++) {
		if(!fread(&c, 1, 1, flabels)) {
			perror("fread5");
			return -1;
		}

		output->images[i].class = (int) c;
    }

    fclose(flabels);

    return 0;
}

void dataset_free(dataset *dt)
{
	if(!dt)
		return;

	// TODO: Expliquer pourquoi ça marche
	free(dt->images[0].pixels);
	free(dt->images);

	*dt = empty_set;
}

int main(int argc, char* argv[]) 
{
    size_t i;
	int j, max_index, training_iterations;
	double output[10];
	int confusion_matrix[10 * 10];
    dataset training, tests;

	if(argc != 5) {
		printf("./mnist [NUMBER OF HIDDEN LAYERS] [NEURON PER HIDDEN LAYERS] [TRAINING ITERATION] [OUTPUT FILE]");
		return 1;
	}

	dataset_read(&training,
        "./DATA/train-images-idx3-ubyte",
        "./DATA/train-labels-idx1-ubyte");
    dataset_read(&tests,
        "./DATA/t10k-images-idx3-ubyte",
        "./DATA/t10k-labels-idx1-ubyte");

	assert(training.width == tests.width);
	assert(training.height == tests.height);

    genann *ann = genann_init(training.width * training.height,
		atoi(argv[1]),
		atoi(argv[2]),
		10
	);

	assert(ann != NULL);


    training_iterations = atoi(argv[3]);
	bzero(output, 10 * sizeof(double));
	for(j = 0; j < training_iterations; j ++) {
		for (i = 0; i < training.nimages; ++i) {
			fprintf(stderr, "[%d / %d]: %ld%%\r",
                    j+1,
                    training_iterations,
                    (100 * (i+1)) / training.nimages
            );
			
			output[training.images[i].class] = 1;
			genann_train(ann, training.images[i].pixels, output, 0.3);
			output[training.images[i].class] = 0;
		}
		fprintf(stderr, "\n");
	}

	printf("Writing to %s \n",argv[4]);
	FILE *out = fopen(argv[4], "w");
	genann_write(ann, out);
	fclose(out);

	fprintf(stderr, "Running the tests\n");
	bzero(confusion_matrix, 10 * 10 * sizeof(int));
	for (i = 0; i < tests.nimages; i ++) {
		const double *out = genann_run(ann, tests.images[i].pixels);

		max_index = 0;
		for(j = 1; j < 10; j ++) {
			if(out[j] > out[max_index]) 
				max_index = j;
		}

		confusion_matrix[10 * max_index + tests.images[i].class] ++;

		fprintf(stderr, "Tests: %ld%%\r", (100 * (i+1)) / tests.nimages);
	}
	fprintf(stderr, "\n");

	for(j = 0; j < 10; j ++) {
		for(i = 0; i < 10; i ++) {
			printf("%d,", confusion_matrix[10 * j + i]);
		}
		printf("\n");
	}

	genann_free(ann);

	dataset_free(&training);
	dataset_free(&tests);

	return 0;
}

