#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config.h"
#include "mnist_db.h"
#include "utils.h"

int mnist_init(MnistDataset *output,
	const char *images_file,
	const char *labels_file,
	int shuffle, int transpose,
	size_t batch_size
)
{
	size_t i;
	double *buf;

	if(!output)
		return -1;

	memset(output, 0, sizeof(MnistDataset));
	output->shuffle = shuffle;
	output->transpose = transpose;

#ifndef _MSC_VER
	output->fimage = fopen(images_file, "r");

    if(!output->fimage) {
        perror("fopen");
		return -1;
    }
#else
	if(fopen_s(&output->fimage, images_file, "rb"))
		return 1;
#endif

#ifndef _MSC_VER
	output->flabel = fopen(labels_file, "r");

    if(!output->flabel) {
        perror("fopen");
		return -1;
    }
#else
	if(fopen_s(&output->flabel, labels_file, "rb"))
		return 1;
#endif

	fseek(output->fimage, 4, SEEK_SET);

    if(!fread(&output->entries_count, 4, 1, output->fimage)) {
        perror("fread1");
		return -1;
    }
	
    if(!fread(&output->width, 4, 1, output->fimage)) {
        perror("fread2");
		return -1;
    }

    if(!fread(&output->height, 4, 1, output->fimage)) {
        perror("fread3");
		return -1;
    }

#ifdef LITTLE_ENDIAN
    output->entries_count = CHANGE_ENDIANNESS(output->entries_count);
    output->width  = CHANGE_ENDIANNESS(output->width);
    output->height = CHANGE_ENDIANNESS(output->height);
#endif /* LITTLE_ENDIAN */

	if(batch_size != 0)
		output->batch_size = batch_size;
	else
		output->batch_size = output->entries_count;

	printf("Batch size: %zd; Width: %d; Height: %d\n",
		output->batch_size, output->width, output->height);

    output->batch_entries = malloc(sizeof(MnistEntry) * output->batch_size);
    if(!output->batch_entries) {
        perror("malloc");
		return -1;
    }

	buf = malloc(sizeof(double) * output->width * output->height * output->batch_size);
	if(!buf) {
		perror("malloc");
		return -1;
	}

	for(i = 0; i < output->batch_size; i ++) {
		output->batch_entries[i].class = 0;
		output->batch_entries[i].pixels = buf + i * (output->width * output->height);
	}

	output->entries_to_read = malloc(sizeof(int) * output->entries_count);
	if(!output->entries_to_read) {
		perror("malloc");
		return -1;
	}

	for(i = 0; i < output->entries_count; i ++)
		output->entries_to_read[i] = i;

    return 0;
}

size_t mnist_load_batch(MnistDataset *dt)
{
	size_t i, j;
	size_t x, y;
	double tmp;
	MnistEntry *entry;
	const size_t MNIST_ENTRY_SIZE = dt->width * dt->height;
	unsigned char buf[MNIST_ENTRY_SIZE + 1];

	if(dt->entries_read >= dt->entries_count)
		dt->entries_read = 0;

	if(dt->entries_read == 0 && dt->shuffle)
		shuffle(dt->entries_to_read, dt->entries_count);

	for(i = 0; i < dt->batch_size; i ++, dt->entries_read ++) {
		entry = &dt->batch_entries[i];

		if(dt->entries_read >= dt->entries_count)
			break;

		fseek(dt->fimage, MNIST_ENTRY_SIZE * dt->entries_to_read[dt->entries_read] + 16, SEEK_SET);
		fseek(dt->flabel, dt->entries_to_read[dt->entries_read] + 8, SEEK_SET);

		/* Lit l'étiquette */
		if(!fread(buf, 1, 1, dt->flabel)) {
			perror("fread");
			break;
		}
		entry->class = (int) buf[0];

		/* Lit l'image */
		if(!fread(buf, MNIST_ENTRY_SIZE, 1, dt->fimage)) {
			perror("fread");
			break;
		}

		for(j = 0; j < MNIST_ENTRY_SIZE; j ++)
			entry->pixels[j] = ((double) buf[j]) / 255.;
	}

	if(dt->noise) {
		for(i = 0; i < dt->batch_size; i ++) {
			entry = &dt->batch_entries[i];
			for(j = 0; j < MNIST_ENTRY_SIZE; j ++)
				entry->pixels[j] += ((double) rand() / (double) RAND_MAX) * dt->noise_amplitude;
		}
	}

	if(dt->transpose) {
		for(i = 0; i < dt->batch_size; i ++) {
			entry = &dt->batch_entries[i];
			for(x = 0; x < dt->width; x ++) {
				for(y = x+1; y < dt->height; y ++) {
					/* Swap entry->pixels[x + dt->width * y] and entry->pixels[y + dt->width * x] */
					tmp = entry->pixels[x + dt->width * y];
					entry->pixels[x + dt->width * y] = entry->pixels[y + dt->width * x];
					entry->pixels[y + dt->width * x] = tmp;
				}	
			}
		}
	}

	return i;
}

void mnist_add_noise(MnistDataset *dt, double noise_amplitude)
{
	dt->noise = 1;
	dt->noise_amplitude = noise_amplitude;
}

void mnist_free(MnistDataset *dt)
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
		MnistEntry, d'où ce free en particulier.
	*/
    fclose(dt->fimage);
    fclose(dt->flabel);

	free(dt->batch_entries[0].pixels);
	free(dt->batch_entries);

	free(dt->entries_to_read);

	memset(dt, 0, sizeof(MnistDataset));
}
