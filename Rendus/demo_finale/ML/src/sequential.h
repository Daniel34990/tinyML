#ifndef _SEQUENTIAL_H_
#define _SEQUENTIAL_H_

#include <stddef.h>

#include "genann.h"
#include "pooling.h"
#include "convolution.h"
#include "activation.h"
#include "flatten.h"
#include "tensor.h"

enum SequentialLayerType {
	SEQUENTIAL_LAYER_NONE = 0,
	SEQUENTIAL_LAYER_DENSE,
	SEQUENTIAL_LAYER_CONV_2D,
	SEQUENTIAL_LAYER_POOLING_2D,
	SEQUENTIAL_LAYER_ACTIVATION_2D,
	SEQUENTIAL_LAYER_FLATTEN
};
typedef enum SequentialLayerType SequentialLayerType;

typedef struct SequentialLayer SequentialLayer;
struct SequentialLayer {
	SequentialLayerType type;
	void *elt;
};

typedef struct Sequential Sequential;
struct Sequential {
	SequentialLayer *layers;
	size_t layer_count;

	size_t input_width;
	size_t input_height;
};

typedef struct Sequential_Actfun Sequential_Actfun;
struct Sequential_Actfun {
	actfun fun;
	derivate_actfun derivate;
};

/* Initialise une séquence de couche */
Sequential* sequential_create(size_t input_width, size_t input_height, size_t layer_count);

/* Libère le sequential et toute ses couches */
void sequential_free(Sequential*);

/* Défini la couche i comme étant une couche de pooling */
void sequential_set_pooling_2d(Sequential *, size_t i, PoolingLayer *);

/* Défini la couche i comme étant un réseau de neurones denses */
void sequential_set_dense(Sequential *, size_t i, genann const *);

/* Défini la couche i comme étant une couche de convolution */
void sequential_set_convolution(Sequential *, size_t i, ConvolutionLayer const *);

/* Défini la couche i comme étant une couche d'activation */
void sequential_set_activation(Sequential *, size_t i, ActivationLayer const *);

/*
 Défini la couche i comme étant une couche
 transformant les dimensions de son entrée.
*/
void sequential_set_flatten(Sequential *, size_t i, FlatteningLayer const *);

/* Execute les couches définis */
Tensor sequential_run(Sequential const *seq, Tensor input);

/* Entraine toute les couches */
void sequential_train(Sequential const *seq, Tensor input, Tensor desired_output, double learning_rate);

/* Enregistre le modèle */
void sequential_write(Sequential const *seq, FILE *out);

/* Lis un modèle d'un fichier */
Sequential* sequential_read(FILE *in, Sequential_Actfun(*actfun_finder)(const char*));

#endif /* _SEQUENTIAL_H_ */
