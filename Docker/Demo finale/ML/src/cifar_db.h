#ifndef _CIFAR_DB_H_
#define _CIFAR_DB_H_

#include <stdio.h>

#define CIFAR_IMAGE_SIZE	32

typedef struct CifarEntry CifarEntry;
struct CifarEntry {
	double *red;
	double *green;
	double *blue;

	int label;
};

typedef struct CifarDataset CifarDataset;
struct CifarDataset {
	int shuffle;
	
	size_t batch_size;
	size_t entries_count;
	size_t entries_read;

	FILE *f;
	
	int *entries_to_read;
	CifarEntry *batch_entries;
};

int cifar_init(CifarDataset*, const char *path, int shuffle, size_t batch_size);
int cifar_load_batch(CifarDataset*);
void cifar_free(CifarDataset*);

#endif /* _CIFAR_DB_H_ */