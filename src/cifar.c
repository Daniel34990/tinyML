#include <stdio.h>

#include "cifar_db.h"

int main(void)
{
	int ret;
	CifarDataset dt;

	ret = cifar_init(&dt, "cifar-100-binary/test.bin", 1, 256);
	if(ret)
		return -1;

	cifar_free(&dt);

	return 0;
}