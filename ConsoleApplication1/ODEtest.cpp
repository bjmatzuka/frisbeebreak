/* cvpendulum.c -- pendulum demo
* dbindel, Apr 2009.
*
* Adapted from demo/cvode/serial/cdenx.c
* by Scott Cohen, Alan Hindmarsh, and Radu Serban.
*
* For more info, see CVODE web page:
* https://computation.llnl.gov/casc/sundials/main.html
*/
#include <stdio.h>
#include <armadillo>


// Bindel Scientific Computing(G63.2043.001 / G22.2112.001)
#include <stdlib.h>
#include <math.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <cvode/cvode_dense.h>
#include <sundials/sundials_dense.h>
#include <sundials/sundials_types.h>

using namespace std;
using namespace arma;

void ekf(void* cvode_mem, mat* data, N_Vector* y);
static int f(realtype t, N_Vector y, N_Vector ydot, void *f_data)
{
	realtype theta = NV_Ith_S(y, 0);
	realtype omega = NV_Ith_S(y, 1);
	realtype omegap = -sin(theta);
	NV_Ith_S(ydot, 0) = omega;
	NV_Ith_S(ydot, 1) = omegap;
	return 0;
}
int ODEtest1()
{
	mat data_orig;
	data_orig.zeros(200, 3);
	mat data;
	data.zeros(200, 3);

	int N = 200;
	realtype T0 = 0;
	realtype Tfinal = 10;
	realtype theta0 = 1;
	realtype reltol = 1e-6;
	realtype abstol = 1e-8;
	realtype t;
	int flag, k;
	N_Vector y = NULL;
	void* cvode_mem = NULL;
	/* Create serial vector of length NEQ for I.C. */
	y = N_VNew_Serial(2);

	// Bindel Scientific Computing(G63.2043.001 / G22.2112.001)
	NV_Ith_S(y, 0) = theta0;
	NV_Ith_S(y, 1) = 0;
	/* Set up solver */
	cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
	if (cvode_mem == 0) {
		fprintf(stderr, "Error in CVodeMalloc: could not allocate\n");
		return -1;
	}
	/* Call CVodeMalloc to initialize the integrator memory */
	flag = CVodeInit(cvode_mem, f, T0, y );
	if (flag < 0) {
		fprintf(stderr, "Error in CVodeInit: %d\n", flag);
		return -1;
	}
	flag = CVodeSStolerances(cvode_mem, reltol, abstol);
	if (flag < 0) {
		fprintf(stderr, "Error in CVodeSStolerances: %d\n", flag);
		return -1;
	}

	data_orig(0, 1) = 1;
	/* In loop, call CVode, print results, and test for error. */
	for (k = 1; k < N; ++k) {
		realtype tout = k*Tfinal / N;
		if (CVode(cvode_mem, tout, y, &t, CV_NORMAL) < 0) {
			fprintf(stderr, "Error in CVode: %d\n", flag);
			return -1;
		}
		//printf("%g %.16e %.16e\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1));
		data_orig(k, 0) = t;
		data_orig(k, 1) = NV_Ith_S(y, 0);
		data_orig(k, 2) = NV_Ith_S(y, 1);

		data(k, 0) = t;
		data(k, 1) = data_orig(k, 1) + 0.01*randn(1)[0];
		data(k, 2) = data_orig(k, 2) + 0.01*randn(1)[0];
	}
	data.print("data:");

	NV_Ith_S(y, 0) = theta0;
	NV_Ith_S(y, 1) = 0;

	//return NV_Ith_S(y,0);
	//ekf(cvode_mem, &data, &y);
	N_VDestroy_Serial(y); /* Free y vector */
	CVodeFree(&cvode_mem); /* Free integrator memory */
	return(0);
}