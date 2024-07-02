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

int main(int argc, char *argv[]) {
    size_t i;
	int j, max_index;
    int confusion_matrix[10 * 10];
    dataset training, tests;
    FILE *in = fopen(argv[1], "r");
    genann *ann = genann_read(in);

    dataset_read(&tests,
        "./DATA/t10k-images-idx3-ubyte",
        "./DATA/t10k-labels-idx1-ubyte");

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

    // Normalize the confusion matrix
    int total_samples[10] = {0,0,0,0,0,0,0,0,0,0};
    for(j = 0; j < 10; j ++) {
        for(i = 0; i < 10; i ++) {
            total_samples[j] += confusion_matrix[10 * j + i];
        }
    }

    fprintf(stderr, "Normalized Confusion Matrix:\n");
    for(j = 0; j < 10; j ++) {
        for(i = 0; i < 10; i ++) {
            double normalized_value = (double)confusion_matrix[10 * j + i] / total_samples[j];
            fprintf(stderr, "%.2f,", normalized_value);
        }
        fprintf(stderr,"\n");
    }

	genann_free(ann);
	dataset_free(&tests);

	return 0;

}