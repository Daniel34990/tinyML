#include <stdlib.h>

#include "utils.h"

int max(int a, int b)
{ return a >= b ? a : b; }

void shuffle(int* arr, size_t len)
{
	/* Ceci est une implémentation de Fisher-Yates */
	size_t i, j;
	int tmp;
	if(len < 2)
		return;

	for(i = 0; i <= len-2; i ++) {
		j = i + rand() % (len-i);

		tmp = arr[i];
		arr[i] = arr[j];
		arr[j] = tmp;
	}
}
