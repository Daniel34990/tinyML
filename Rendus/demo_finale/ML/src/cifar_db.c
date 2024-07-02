#include <assert.h>
#include <stdlib.h>

#include "cifar_db.h"
#include "utils.h"

/*
const int CIFAR_LABEL_OFFSET = 0;
const int CIFAR_RED_OFFSET = 1;
const int CIFAR_GREEN_OFFSET = 1 + 1024;
const int CIFAR_BLUE_OFFSET = 1 + 2 * 1024;
*/

const int CIFAR_ENTRY_SIZE = 1 + 3 * CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE;

int cifar_init(CifarDataset* output, const char *path, int shuffle, size_t batch_size)
{
	size_t i;
	CifarEntry *entry;
	double *color_buf;

	assert(path);
	assert(output);

	output->f = fopen(path, "r");
	if(!output->f) {
		perror("fopen");
		return 1;
	}
	output->shuffle = shuffle;

	/* Retrieve the file's size */
	fseek(output->f, 0, SEEK_END);
	output->entries_count = ftell(output->f) / CIFAR_ENTRY_SIZE;
	rewind(output->f);

	if(batch_size == 0)
		output->batch_size = output->entries_count;
	else
		output->batch_size = batch_size;

	/* Allocate the entries */
	output->batch_entries = (CifarEntry*) malloc(sizeof(CifarEntry) * output->batch_size);
	if(!output->batch_entries) {
		perror("malloc");
		return 1;
	}

	/* Prepare the batch */
	color_buf = malloc(sizeof(double)
		* CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE
		* 3 * output->batch_size);
	if(!color_buf) {
		perror("malloc");
		return 1;
	}

	for(i = 0; i < output->batch_size; i ++) {
		entry = &output->batch_entries[i];
		entry->red   = color_buf + CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE * (3*i + 0);
		entry->green = color_buf + CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE * (3*i + 1);
		entry->blue  = color_buf + CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE * (3*i + 2);
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

void cifar_free(CifarDataset* db)
{
	free(db->batch_entries[0].red);
	free(db->batch_entries);
	free(db->entries_to_read);
	fclose(db->f);
}

int cifar_load_batch(CifarDataset* dt)
{
	size_t i, j;
	char buf[CIFAR_IMAGE_SIZE * CIFAR_IMAGE_SIZE];
	CifarEntry *entry;

	if(dt->entries_read >= dt->entries_count)
		dt->entries_read = 0;

	if(dt->entries_read == 0 && dt->shuffle)
		shuffle(dt->entries_to_read, dt->entries_count);

	for(i = 0; i < dt->batch_size; i ++, dt->entries_read ++) {
		entry = &dt->batch_entries[i];

		/* Choisi l'entrée à lire */
		fseek(dt->f,
			CIFAR_ENTRY_SIZE * dt->entries_to_read[dt->entries_read],
			SEEK_SET);

		/* Lit l'étiquette buf[0] */
		if(!fread(buf, 1, 1, dt->f)) {
			perror("fread");
			break;
		}

		entry->label = (int) buf[0];

		/* Lit le rouge */
		if(fread(buf, 1, CIFAR_IMAGE_SIZE*CIFAR_IMAGE_SIZE, dt->f) != CIFAR_IMAGE_SIZE*CIFAR_IMAGE_SIZE) {
			perror("fread");
			break;
		}

		for(j = 0; j < CIFAR_IMAGE_SIZE*CIFAR_IMAGE_SIZE; j ++)
			entry->red[j] = ((double) buf[j]) / 255.;

		/* Lit le vert */
		if(fread(buf, 1, CIFAR_IMAGE_SIZE*CIFAR_IMAGE_SIZE, dt->f) != CIFAR_IMAGE_SIZE*CIFAR_IMAGE_SIZE) {
			perror("fread");
			break;
		}

		for(j = 0; j < CIFAR_IMAGE_SIZE*CIFAR_IMAGE_SIZE; j ++)
			entry->green[j] = ((double) buf[j]) / 255.;

		/* Lit le bleu */
		if(fread(buf, 1, CIFAR_IMAGE_SIZE*CIFAR_IMAGE_SIZE, dt->f) != CIFAR_IMAGE_SIZE*CIFAR_IMAGE_SIZE) {
			perror("fread");
			break;
		}

		for(j = 0; j < CIFAR_IMAGE_SIZE*CIFAR_IMAGE_SIZE; j ++)
			entry->blue[j] = ((double) buf[j]) / 255.;
	}

	return i;
}