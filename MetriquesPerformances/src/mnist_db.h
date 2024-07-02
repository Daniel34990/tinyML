#ifndef _MNIST_DB_H_
#define _MNIST_DB_H_

#include <stddef.h>

#define CHANGE_ENDIANNESS(a)			\
	( 									\
		(((a) >>  0) & 0xFF) << 24 |	\
        (((a) >>  8) & 0xFF) << 16 |	\
        (((a) >> 16) & 0xFF) <<  8 |	\
        (((a) >> 24) & 0xFF) <<  0 		\
	)

typedef struct image image;
struct image {
    int class;
    double *pixels;
};

typedef struct dataset dataset;
struct dataset {
    size_t nimages;
    unsigned int width;
	unsigned int height;
    image *images;
};

#define CLASS_COUNT 10

int dataset_read(dataset *output, char *images_file, char *labels_file);
void dataset_free(dataset *dt);

#endif /* _MNIST_DB_H_ */
