#include "stdafx.h"
#include <iostream>
#include <armadillo>

#include <stdlib.h>
#include <math.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <cvode/cvode_dense.h>
#include <sundials/sundials_dense.h>
#include <sundials/sundials_types.h>

using namespace std;
using namespace arma;

void ekf(void* cvode_mem, mat* data, N_Vector* y) {
	int Nrowdata = data->n_rows;
	realtype t;
	int flag;
	/* In loop, call CVode, print results, and test for error. */
	for (int k = 0; k < Nrowdata; k++) {
		realtype tout = (*data)(0, k);
		if (CVode(cvode_mem, tout, *y, &t, CV_NORMAL) < 0) {
			fprintf(stderr, "Error in CVode: %d\n", flag);
			return;
		}
		// printf("%g %.16e %.16e\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1));
	}

}