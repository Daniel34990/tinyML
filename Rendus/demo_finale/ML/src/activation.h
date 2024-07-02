#ifndef _ACTIVATION_H_
#define _ACTIVATION_H_

#include <stddef.h>

#include "tensor.h"

typedef double (*actfun)(void*, double);
typedef double (*derivate_actfun)(void*, double, double);

typedef struct ActivationLayer ActivationLayer;
struct ActivationLayer {
	int last_input_tainted;

	Tensor last_input;
	Tensor last_output;

	actfun activation;
	derivate_actfun derivate_act;
	char* actfun_name;
};

/* Initialise la couche d'activation */
ActivationLayer* activation_init(char* actfun_name,
	actfun activation, derivate_actfun derivate_act,
	size_t input_width, size_t input_height, size_t input_depth);

/* Libère la couche d'activation de la mémoire */
void activation_free(ActivationLayer*);

/* Applique la couche d'activation sur l'entrée inputs. */
Tensor activation_run(ActivationLayer *layer, Tensor inputs);

/* Applique le gradient descent sur la matrice backprop. */
Tensor activation_backpropagate(ActivationLayer *layer, Tensor backprop);

#endif /* _ACTIVATION_H_ */
