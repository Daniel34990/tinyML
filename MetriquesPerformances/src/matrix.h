#ifndef _MATRIX_H_
#define _MATRIX_H_

#include <stddef.h>

typedef struct Matrix Matrix;
struct Matrix {
	size_t width;
	size_t height;

	double *data;
};

void matrix_new(Matrix* out, size_t width, size_t height);
void matrix_create_output_convolution(Matrix *out, Matrix filter, unsigned int stride, Matrix input);
void matrix_convolution(Matrix *out, Matrix filter, unsigned int stride, Matrix input);
double matrix_trace(Matrix m);

#endif /* _MATRIX_H_ */
