#ifndef MESURES_H
#define MESURES_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double categorical_crossentropy(genann* net,dataset tests);
void fit_model(dataset training, dataset tests, int hidden_layers, int hidden, FILE *fp);
dataset reduced_dataset(dataset original_dataset, int nimages);
int is_new_ai_better(genann *old, genann *new, dataset tests);
double accuracy(genann* ann, dataset tests);

#endif /* MESURES_H */
