#ifndef _FLATTEN_H_
#define _FLATTEN_H_

#include <stddef.h>

#include "tensor.h"

typedef struct FlatteningLayer FlatteningLayer;
struct FlatteningLayer {
	size_t input_width;
	size_t input_height;
	size_t input_depth;
};

/*  */
FlatteningLayer *flatten_init(size_t input_width, size_t input_height, size_t input_depth);

/*  */
Tensor flatten_run(FlatteningLayer *, Tensor input);

/*  */
Tensor flatten_backpropagate(FlatteningLayer *, Tensor bt_input);

#endif /* _FLATTEN_H_ */
