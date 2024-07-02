#ifndef _TENSOR_H_
#define _TENSOR_H_

#include <stddef.h>
#include <stdio.h>

typedef struct Tensor Tensor;
struct Tensor {
	size_t width;
	size_t height;
	size_t depth;

	double *data;
};

/*
 Crée un tenseur de taille "width" * "height" * "depth".
 Renvoie 1 en cas d'erreur, 0 sinon
*/
int tensor_new(Tensor* out, size_t width, size_t height, size_t depth);

/* Libère le tenseur m de la mémoire */
void tensor_free(const Tensor m);

/* Calcule la trace du tenseur m */
double tensor_trace(const Tensor m);

/* tensor_rotate_180 */
void tensor_rotate_180(Tensor m);

/* Initialise les coefficients du tenseur m à 0. */
void tensor_clear(const Tensor m);

/* Ajoute un à un les coefficients de rhs dans le tenseur "out" après
   les avoir multipliés par c. */
void tensor_scale_and_add(Tensor m, const Tensor rhs, double c);

/*
 Initialise les coefficients du tenseur
 avec des valeurs aléatoires entre -1 et 1.
*/
void tensor_randomize(const Tensor *m);

/*
 Calcule la convolution entre la matrice "input", doté d'un padding de 
 "pad_width" x "pad_height", avec le filtre "filter" (une matrice),
 et additionne le résultat aux coefficients du tenseur "out".
*/
void tensor_convolution_add(Tensor out,
	Tensor filter, unsigned int stride,
	Tensor input,
	unsigned int pad_width, unsigned int pad_height);

/* Ecrit le contenu d'un tenseur dans un fichier */
void tensor_write(Tensor, FILE*);

/* Lit le contenu d'un tenseur dans un fichier */
Tensor tensor_read(Tensor, FILE*);

/* Copie le contenu d'un tenseur dans une autre */
void tensor_copy(Tensor*, const Tensor);

/* Renvoie le nombre de valeurs contenu dans un tenseur */
size_t tensor_size(Tensor);

/* Renvoie la sous-matrice à l'indice i */
Tensor tensor_get_matrix(Tensor t, size_t i);

#endif /* _TENSOR_H_ */
