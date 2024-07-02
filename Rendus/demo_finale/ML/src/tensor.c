#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <stdio.h>

#include "tensor.h"
#include "utils.h"

int tensor_new(Tensor* out, size_t width, size_t height, size_t depth)
{
	assert(out);
	assert(width > 0);
	assert(height > 0);
	assert(depth > 0);

	out->data = malloc(width * height * depth * sizeof(double));
    if(!out->data) {
        perror("malloc");
		bzero(out, sizeof(Tensor));
        return 1;
    }

	out->width = width;
	out->height = height;
	out->depth = depth;
	tensor_clear(*out);
    return 0;
}

void tensor_convolution_add(Tensor out,
	Tensor filter, unsigned int stride,
	Tensor input,
	unsigned int pad_width, unsigned int pad_height)
{
	size_t i, j, k;
	int x, y;
	int x0, y0;
	double tmp;

	assert(out.depth == 1);
	assert(input.depth == filter.depth);
	//assert(filter.depth == 1);

	assert(out.width == (input.width - filter.width + 2*pad_width) / stride + 1);
	assert(out.height == (input.height - filter.height + 2*pad_height) / stride + 1);

	for(j = 0; j < out.height; j ++) {
		y0 = j * stride - pad_height;

		for(i = 0; i < out.width; i ++) {
			x0 = i * stride - pad_width;

			tmp = 0.;

			for(y = max(-y0, 0); y < (int) filter.height; y ++) {
				if((y + y0) >= (int) input.height)
					break;
				
				for(x = max(-x0, 0); x < (int) filter.width; x ++) {
					if((x + x0) >= (int) input.width)
						break;

					for(k = 0; k < filter.depth; k ++) {
						tmp +=
							filter.data[(k * filter.height + y) * filter.width + x] *
							input.data[(k * input.height + y + y0) * input.width + (x + x0)];
					}
				}
			}

			out.data[j * out.width + i] += tmp;
		}
	}
}

double tensor_trace(Tensor m)
{
	double out;
	size_t k;

	assert(m.data);
	assert(m.depth == 1);
	assert(m.height == m.width);

	out = 0;
	for(k = 0; k < m.height; k ++)
		out += m.data[k * m.width + k];

	return out;
}

void tensor_rotate_180(Tensor m)
{
	size_t i;
	size_t start_index, end_index;

	const size_t elt_per_matrix = m.height * m.width;

    assert(m.data);

	for(i = 0; i < m.depth; i ++) {

		start_index = i*elt_per_matrix;
		end_index = start_index + elt_per_matrix - 1;
		
		while (start_index < end_index) {
			double temp = m.data[start_index];
			m.data[start_index] = m.data[end_index];
			m.data[end_index] = temp;
			
			start_index ++;
			end_index --;
		}
	}
}

void tensor_free(const Tensor m)
{
	free(m.data);
}

void tensor_scale_and_add(Tensor out, const Tensor rhs, double c)
{
	size_t i;

	assert(out.depth == rhs.depth);
	assert(out.width == rhs.width);
	assert(out.height == rhs.height);

	for(i = 0; i < out.width * out.height * out.depth; i ++)
		out.data[i] += rhs.data[i] * c;
}

void tensor_clear(const Tensor m)
{
	bzero(m.data, m.width * m.height * m.depth * sizeof(double));
}

void tensor_randomize(const Tensor *m)
{
    size_t i;
    for(i = 0; i < m->width * m->height * m->depth; i++) {
        m->data[i] = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    }
}

void tensor_write(Tensor m, FILE* out)
{
	size_t k;

	fprintf(out, "%zu %zu %zu\n", m.width, m.height, m.depth);
	for(k = 0; k < m.width * m.height * m.depth; k ++)
		fprintf(out, "%lf ", m.data[k]);
	fputc('\n', out);
}

Tensor tensor_read(Tensor out, FILE* in)
{
	size_t k;
	size_t width, height, depth;

	fscanf(in, "%zu %zu %zu\n", &width, &height, &depth);
	assert(out.width == width);
	assert(out.height == height);
	assert(out.depth == depth);

	for(k = 0; k < width * height * depth; k ++)
		fscanf(in, "%lf", &out.data[k]);

	while(fgetc(in) != '\n');

	return out;
}

void tensor_copy(Tensor* dst, const Tensor src)
{
	assert(dst->width == src.width);
	assert(dst->height == src.height);
	assert(dst->depth == src.depth);

	memcpy(dst->data, src.data, src.width * src.height * src.depth * sizeof(double));
}

size_t tensor_size(Tensor t)
{ return t.width * t.height * t.depth; }

Tensor tensor_get_matrix(Tensor t, size_t i)
{
	Tensor output;

	assert(i < t.depth);

	output.width = t.width;
	output.height = t.height;
	output.depth = 1;

	output.data = t.data + i * (t.width * t.height);

	return output;
}