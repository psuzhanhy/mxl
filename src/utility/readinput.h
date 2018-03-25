#ifndef READINPUT_H
#define READINPUT_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include "matvec.h"

CSR_matrix Dense2CSR (DenseData denseInput);
void ReadDenseInput (char *DataFile, DenseData *data); 

#endif
