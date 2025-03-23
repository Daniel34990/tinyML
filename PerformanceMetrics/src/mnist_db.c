#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "mnist_db.h"

int dataset_read(dataset *output, char *images_file, char *labels_file)
{
	char c;
	size_t i;
    FILE *fimage;

	if(!output)
		return -1;

	memset(output, 0, sizeof(dataset));

#ifndef _MSC_VER
	fimage = fopen(images_file, "r");

    if(!fimage) {
        perror("fopen");
		return -1;
    }
#else
	if(fopen_s(&fimage, images_file, "rb"))
		return 1;
#endif

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

#ifdef LITTLE_ENDIAN
    output->nimages = CHANGE_ENDIANNESS(output->nimages);
    output->width  = CHANGE_ENDIANNESS(output->width);
    output->height = CHANGE_ENDIANNESS(output->height);
#endif /* LITTLE_ENDIAN */

	printf("Image count: %zd; Width: %d; Height: %d\n",
		output->nimages, output->width, output->height);

    output->images = malloc(sizeof(image) * output->nimages);
    if(!output->images) {
        perror("malloc");
		return -1;
    }

	unsigned char* raw_buffer = malloc(sizeof(unsigned char) * output->width * output->height * output->nimages);
	i = fread(raw_buffer, sizeof(unsigned char) * output->width * output->height, output->nimages, fimage);
	if(i != output->nimages) {
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

	/*
		Cette ligne de code fonctionne, car
		elle repose sur le fait que les pixels
		des différentes images soient sur un
		même buffer contigue, et que l'addresse
		du début de ce dit buffer correspond
		à l'addresse du début de la première
		image, d'où ce free en particulier.
	*/
	free(dt->images[0].pixels);
	free(dt->images);

	memset(dt, 0, sizeof(dataset));
}
