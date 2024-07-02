#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "matrix.h"

void matrix_new(Matrix* out, size_t width, size_t height)
{
	assert(out);
	assert(width > 0);
	assert(height > 0);

	out->width = width;
	out->height = height;
	out->data = malloc(out->width * out->height * sizeof(double));
	assert(out->data);
	bzero(out->data, out->width * out->height * sizeof(double));
}

void matrix_create_output_convolution(Matrix *out, Matrix filter, unsigned int stride, Matrix input)
{
	assert(stride > 0);
	assert(filter.width <= input.width);
	assert(filter.height <= input.height);

	matrix_new(out,
		(input.width  - filter.width)  / stride,
		(input.height - filter.height) / stride
	);
}

void matrix_convolution(Matrix *out, Matrix filter, unsigned int stride, Matrix input)
{
	size_t i, j;
	size_t x, y;
	size_t x0, y0;

	for(j = 0; j < out->height; j ++) {
		for(i = 0; i < out->width; i ++) {
			out->data[j * out->width + i] = 0;

			x0 = i * stride;
			y0 = j * stride;
			for(y = 0; y < filter.height; y ++) {
				for(x = 0; x < filter.width; x ++) {
					out->data[j * out->width + i] +=
						filter.data[y * filter.width + x] *
						input.data[(y + y0) * input.width + (x + x0)];
				}
			}
		}
	}
}

double matrix_trace(Matrix m)
{
	double out;
	size_t k;

	assert(m.data);
	assert(m.height == m.width);

	out = 0;
	for(k = 0; k < m.height; k ++)
		out += m.data[k * m.width + k];

	return out;
}