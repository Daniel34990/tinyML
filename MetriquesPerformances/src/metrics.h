#ifndef METRICS_H
#define METRICS_H

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

double categorical_crossentropy(genann* net,dataset tests);
void fit_model(dataset training, dataset tests, int hidden_layers, int hidden);
dataset reduced_dataset(dataset original_dataset, int nimages);
int is_new_ai_better(genann *old, genann *new, dataset tests);
void conf_data(dataset training, dataset tests,int hidden_layers, int hidden,FILE* fp);

#endif /* METRICS_H */

