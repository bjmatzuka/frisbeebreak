#include "stdafx.h"
#include <iostream>
#include <armadillo>
//#include <Rcpp.h>

#include <stdlib.h>
#include <math.h>
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <cvode/cvode_dense.h>
#include <sundials/sundials_dense.h>
#include <sundials/sundials_types.h>
#include "UKF.h"

using namespace std;
using namespace arma;
//using namespace Rcpp;

// this needs to have R function inputs
int (*fpointer)(double*, double*, double*);

static int forig(realtype t, N_Vector y, N_Vector ydot, void *f_data)
{
	realtype theta = NV_Ith_S(y, 0);
	realtype omega = NV_Ith_S(y, 1);
	realtype omegap = -sin(theta);
	NV_Ith_S(ydot, 0) = omega;
	NV_Ith_S(ydot, 1) = omegap;
	return 0;
}

static int f_rtest(double* t, double* y, double* ydot)
{
	ydot[0] = y[1];
	ydot[1] = -sin(y[0]);
	return 0;
}

static int f_wrap(realtype t, N_Vector y, N_Vector ydot, void *f_data)
{
	// this will unpack the CVode inputs and convert them to Cpp.

	//fpointer(&t, (double*) y->content, (double*) ydot->content);
	fpointer(&t, (double*)(N_VGetArrayPointer(y)), (double*)(N_VGetArrayPointer(ydot)));
	return 0;
}

void UTpred(mat x, mat PP, vec time, UTdataOut *result) {
	// unscented tuning parameters: a, b, and k
	//
	// alpha can be 1<= alpha <= 0.0001
	// kappa can be 0 or 3 - state dimension (n)
	// beta is set to be 2 for gaussian

	// this is purely for testing purposes
	//
	fpointer = &f_rtest;

	double aa;
	double bb;
	double kk;
	aa = 1;
	bb = 2;
	kk = 0;

	// initialize unscented transform parameters
	double n;
	double lambda;
	n = as_scalar(x.n_elem);
	lambda = aa*aa*(n + kk) - n;

	// initialize matrix square root variables   //

	cx_mat eigvecP;
	cx_vec eigvalP;
	eig_gen(eigvalP, eigvecP, PP);
	mat cc = zeros<mat>(PP.n_rows, PP.n_cols);
	///////////////////////////////////////////////
	// calculate matrix square root              //
	// cc: matrix square root P => P = cc*cc'    //
	///////////////////////////////////////////////
	if (any(eigvalP) < 0)
	{
		cc = real(sqrtmat(PP));
	}
	else {
		// alternate method for calculating matrix square root //
		//
		/////////////////////////////////////////////////////////

		mat U;
		vec s;
		mat V;
		svd(U, s, V, PP);
		mat sqrtD = sqrt(diagmat(s));
		cc = (U*sqrtD)*trans(U);

		// cholesky factorization      //
		//cc = trans(chol(PP));
	}

	// initilize more unscented parameters     //
	//
	// paran = sqrt(n + lambda)                //
	double den;
	double paran;
	mat rootP(cc.n_rows, cc.n_cols);
	den = n + lambda;
	paran = sqrt(den);
	rootP = cc*paran;
	//rootP.print("root P:");

	// calculates the sigma points, Xi = xhat +/- paran*c
	mat Y = repmat(x, 1, n);
	mat XiP = join_horiz(Y - rootP, Y + rootP);
	mat Xi = join_horiz(x, XiP);
	//Xi.print("Xi compare:");

	// creates weighting vectors for unscented mean and covaraince  //
	mat Wm(1, 2 * n + 1);
	mat Wc(1, 2 * n + 1);
	Wm.fill(1 / (2 * (n + lambda)));
	Wc.fill(1 / (2 * (n + lambda)));
	Wm(0, 0) = lambda / (n + lambda);
	Wc(0, 0) = lambda / (n + lambda) + (1 - aa*aa + bb);

	// Weight matrix for transformed covariance calcs
	mat WW;
	mat WmMat = repmat(trans(Wm), 1, Wm.n_elem);
	mat eyeL = eye<mat>(2 * n + 1, 2 * n + 1);
	WW = (eyeL - WmMat)*diagmat(Wc)*trans(eyeL - WmMat);
	//WmMat.print("Wm matrix:");
	//WW.print("W matrix:");

	// integrator results storage
	mat Yi = zeros(n, Xi.n_cols);
	// mean prediction
	mat xt = zeros(n, 1);
	// covariance, Pt
	mat Pt = zeros(n);
	mat Ptc = Pt;

	// time vector decomposition into T0 and Tfinal   //
	realtype T0;
	realtype Tfinal;
	T0 = time(0);
	Tfinal = time(1);

	//cout << "T0" << T0 << endl;
	//cout << "Tfinal" << Tfinal << endl;

	int Num;
	Num = 20;

	// maybe not necessary, but here for now   //
	mat data_orig_new;
	data_orig_new.zeros(Num + 1, 3);
	mat data_new;
	data_new.zeros(Num + 1, 3);

	// set initial value for data_orig and data_new
	data_orig_new(0, 0) = T0;
	data_new(0, 0) = T0;

	// for loop over the sigma points
	int ii;
	for (ii = 0; ii < Xi.n_cols; ++ii) {

		// integrator intialization
		//int Num = 200;
		//realtype T0 = 0;
		//realtype Tfinal = 10;
		//realtype theta0 = 1;
		realtype reltol = 1e-6;
		realtype abstol = 1e-8;
		realtype t;
		int flag, k;
		N_Vector y = NULL;
		void* cvode_mem = NULL;
		/* Create serial vector of length NEQ for I.C. */
		y = N_VNew_Serial(n);

		//NV_Ith_S(y, 0) = theta0;
		//NV_Ith_S(y, 1) = 0;
		int jj;
		for (jj = 0; jj < Xi.n_rows; ++jj) {
			NV_Ith_S(y, jj) = Xi(jj, ii);
			data_orig_new(0, jj + 1) = Xi(jj, ii);
		}
		//cout << NV_Ith_S(y, 0) << "IC 1" << endl;
		//cout << NV_Ith_S(y, 1) << "IC 2" << endl;
		//Xi.print("IC Match");

		/* Set up solver */
		cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
		if (cvode_mem == 0) {
			fprintf(stderr, "Error in CVodeMalloc: could not allocate\n");
			//return -1;
		}
		/* Call CVodeMalloc to initialize the integrator memory */
		flag = CVodeInit(cvode_mem, f_wrap, T0, y);
		if (flag < 0) {
			fprintf(stderr, "Error in CVodeInit: %d\n", flag);
			//return -1;
		}
		flag = CVodeSStolerances(cvode_mem, reltol, abstol);
		if (flag < 0) {
			fprintf(stderr, "Error in CVodeSStolerances: %d\n", flag);
			//return -1;
		}
		//flag = CVodeSetStopTime(cvode_mem, Tfinal);
		//if (flag < 0) {
		//	fprintf(stderr, "Error in CVodeSetStopTime: %d\n", flag);
		//}
		//while (t<Tfinal){
		//	flag = CVode(cvode_mem, Tfinal, y, &t, CV_NORMAL);
		//	if (flag < 0) {
		//				fprintf(stderr, "Error in CVode: %d\n", flag);
		//return -1;
		//		}
		//data_orig_new(0, 1) = 1;
		t = T0;
		for (k = 1; k <= Num; ++k) {
			realtype tout = T0 + (k*(Tfinal - T0) / Num);
			//cout << "time out" << tout << endl;
			if (CVode(cvode_mem, tout, y, &t, CV_NORMAL) < 0) {
				fprintf(stderr, "Error in CVode: %d\n", flag);
				//return -1;
			}
			//printf("%g %.16e %.16e\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1));
			//t.print("Time");

			data_orig_new(k, 0) = t;
			data_orig_new(k, 1) = NV_Ith_S(y, 0);
			data_orig_new(k, 2) = NV_Ith_S(y, 1);

			//data(k, 0) = t;
			//data(k, 1) = data_orig(k, 1) + 0.01*randn(1)[0];
			//data(k, 2) = data_orig(k, 2) + 0.01*randn(1)[0];

		}
		// end of ODE integration
		N_VDestroy_Serial(y); /* Free y vector */
		CVodeFree(&cvode_mem); /* Free integrator memory */
							   //data_orig_new.print("check");

							   // store state at final integrator time
		Yi.col(ii) = trans(data_orig_new(Num, span(1, n)));
		// calculate mean prediction
		xt = xt + Yi.col(ii)*Wm(ii);

	}
	// end of sigma point loop
	//data_orig_new.print("output:");
	//Yi.print("Yi:");
	//xt.print("xt:");
	//covariance and cross covariance calculation
	Pt = Yi*WW*trans(Yi);
	Ptc = Xi*WW*trans(Yi);

	result->yt = xt;
	result->yPt = Pt;
	result->yPtc = Ptc;
}

void UTdata(mat obsf, mat y, mat P, mat obsvec, UTdataOut *result) {
	// set up tuning parameters
	mat alpha(1, 1);
	mat beta(1, 1);
	mat kappa(1, 1);
	alpha(0, 0) = 1;
	beta(0, 0) = 2;
	kappa(0, 0) = 0;

	// initilize unscented transform parameters
	mat n;
	mat lambda;
	n = y.n_elem;
	lambda = as_scalar(alpha)*as_scalar(alpha)*(as_scalar(n) + as_scalar(kappa)) - as_scalar(n);

	//calculates matrix square root
	//c: matrix square root of P => p = c*c'
	mat c = trans(chol(P));

	mat den(1, 1);
	mat paran(1, 1);
	mat rootP(as_scalar(n), as_scalar(n));
	den = as_scalar(n) + as_scalar(lambda);
	paran = sqrt(as_scalar(den));
	rootP = c*as_scalar(paran);

	// create the sigma points, Xi = xhat +/- c*paran
	mat Y = repmat(y, 1, as_scalar(n));
	mat XiP = join_horiz(Y - rootP, Y + rootP);
	mat Xi = join_horiz(y, XiP);

	// creating the weighting vectors and matrices
	mat Wm(1, 2 * as_scalar(n) + 1);
	mat Wc(1, 2 * as_scalar(n) + 1);
	Wm.fill(1 / (2 * (as_scalar(n) + as_scalar(lambda))));
	Wc.fill(1 / (2 * (as_scalar(n) + as_scalar(lambda))));
	Wm(0, 0) = as_scalar(lambda) / (as_scalar(n) + as_scalar(lambda));
	Wc(0, 0) = as_scalar(lambda) / (as_scalar(n) + as_scalar(lambda)) + (1 - as_scalar(alpha)*as_scalar(alpha) + as_scalar(beta));

	// Weight matrix for transformed covariance calcs
	mat WW;
	mat WmMat = repmat(trans(Wm), 1, Wm.n_elem);
	mat eyeL = eye<mat>(2 * as_scalar(n) + 1, 2 * as_scalar(n) + 1);
	WW = (eyeL - WmMat)*diagmat(Wc)*trans(eyeL - WmMat);

	// calculating the statistics of the nonlinear transformation, Y
	// used to calculate the transformed mean, yt
	mat Yi = zeros(obsvec.n_rows, Xi.n_cols);
	mat yt;
	int j;
	for (j = 0; j < Xi.n_cols; ++j) {
		Yi.col(j) = obsf*Xi.col(j);
	}
	// calculates the transformed mean, yt
	yt = Yi*trans(Wm);

	// calculates the transformed covariance and x-covariance
	mat yPt = zeros(as_scalar(n), as_scalar(n));
	mat yPtc = yPt;

	yPt = Yi*WW*trans(Yi);
	yPtc = Xi*WW*trans(Yi);

	result->yt = yt;
	result->yPt = yPt;
	result->yPtc = yPtc;

	//return result;

}

void ukbf(mat obsf, mat data, vec time, vec x0, mat R, mat Q, mat P0, FilterOut *result) {
	// implementation was done using Sarrka(2007)IEEE
	//
	// UKBF Unscented Kalman Bucy Filter
	//
	//  this implements a continuous - discrete unscented kalman filter
	//     INPUTS:
	//      model : rhs of ODE system(include values for parameters)
	//		obser : general function for observations(z = h(x))
	//		data : data points used for filter(column vectors)
	//		time : time period observations occur over
	//          R : covariance noise for data, constant
	//          Q : covariance noise for process, constant
	//          x0 : initial condition for model
	//          P0 : initial condition for covariance
	//          q : parameter values
	//     OUTPUTS :
	//          out.xfilter : filter output
	//          out.time : time scale
	//          out.P : covariance matrices for each time
	//          out.sd : +/ -2 standard deviations

	// initilize the variales
	double L;
	double N;
	//mat xc;
	//cube PC;
	//mat sd;

	L = x0.n_elem;
	N = data.n_rows;
	//xc = zeros(L, N);
	//PC = zeros(L, L, N);
	//sd = zeros(L, N);


	// assign initial values to the respective variables
	result->xfilter.col(0) = x0;
	result->Pfilter.slice(0) = P0;
	result->sdfilter.col(0) = sqrt(diagvec(P0));


	// set up inputs
	vec timevec = zeros(2, 1);
	vec datavec = zeros(data.n_cols, 1);
	UTdataOut statepred;
	UTdataOut stateobs;

	// testing
	//time.print("time");
	//timevec.print("time vec");
	//datavec.print("vector");
	//trans(data.row(1)).print("first data vec");

	// start filter loop
	int i;
	for (i = 1; i < N; ++i) {
		// set up inputs
		timevec(0) = time(i - 1);
		timevec(1) = time(i);
		datavec.col(0) = trans(data.row(i));

		//cout << "iteration" << i << endl;
		//timevec.print("time vec");
		//datavec.print("data vec");
		//xc.col(i - 1).print("xt input");
		//PC.slice(i - 1).print("P input");

		// calculat the unscented transform for the prediction
		//UTpred(xc.col(i - 1), PC.slice(i - 1), timevec, &statepred);
		UTpred(result->xfilter.col(i - 1), result->Pfilter.slice(i - 1), timevec, &statepred);

		//statepred.yt.print("ODE pred");

		// unscented prediction covariance
		mat Pkp = zeros(L, L);
		Pkp = statepred.yPt + Q;

		// calculate the unscented transform for the observation
		UTdata(obsf, statepred.yt, Pkp, datavec, &stateobs);

		//stateobs.yt.print("Obser pred");

		// unscented observation covariance
		mat Sk = zeros(L, L);
		Sk = stateobs.yPt + R;

		// calculate the update step (correction) for the filter
		mat LL;
		mat U;
		//mat U2;
		LL = chol(Sk);
		U = solve(LL, stateobs.yPtc);
		//U2 = trans(solve(trans(LL), trans(result.yPtc)));

		result->Pfilter.slice(i) = Pkp - U*trans(U);
		//adjust = U*(solve(trans(LL), datat - result.yt));
		result->xfilter.col(i) = statepred.yt + U*(solve(trans(LL), datavec - stateobs.yt));
		result->sdfilter.col(i) = sqrt(diagvec(result->Pfilter.slice(i)));

		//xc.col(i).print("x update");
		//PC.slice(i).print("P update");

		//cout << "Iteration:" << i << endl;
		// end of filter loop
	}
	// end of filter function
	//result->xfilter = xc;
	//result->timefilter = time;
	//result->Pfilter = PC;
	//result->sdfilter = sd;
}

void MatMultiply(double* a, double* b, double* c, double* N) {
	mat A = mat(a, *N, *N, false, true);
	mat B = mat(b, *N, *N, false, true);
	mat C = mat(c, *N, *N, false, true);
	vec Cnew;

	C = A*B;

	return;

}

void test_wrapper(double* time, double* y, double* out) {
	*out = (*time)*(*y);
	return;
}

void ukbf_R(double* obsf, double* data, long* M, double* time, long* N, double* x0, long* L, 
	double* R, double* Q, double* P0,
	double* xfilter, double* timefilter, double* Pfilter, double* sdfilter) {
	
	// wrapper that takes R inputs and converts them to C++ armadillo inputs for the C++ functions

	// understanding of pointer dimensions
	// N = length of time/data
	// L = no. of states
	// M = no. of observed states (M<=L)
	fpointer = &f_rtest;

	// converts observation matrix to C++
	mat OBSF = mat(obsf, *L, *L, false, true);
	// converts initial condition to C++
	vec X0 = vec(x0, *L, false, true);
	// converts initial covariance, P0, to C++
	mat PP0 = mat(P0, *L, *L, false, true);
	// converts process noise covariance, R, to C++
	mat R0 = mat(R, *L, *L, false, true);
	// converts obs noise covariance, Q, to C++
	mat Q0 = mat(Q, *M, *M, false, true);
	// converts time to C++
	vec TIME = vec(time, *N, false, true);
	// converts data to C++
	mat DATA = mat(data, *N, *M, false, true);

	//fpointer = &f;


	// output components of filter
	//

	FilterOut fresult;
	// converts filter covariances to C++
	//fresult.Pfilter = cube(Pfilter, *L, *L, *N, false, true);
	fresult.Pfilter = cube(*L,*L,*N);
	// converts filter means to C++
	//fresult.xfilter = mat(xfilter, *L, *N, false, true);
	fresult.xfilter = mat(*L,*N);
	// converts filter standard deviation to C++
	//fresult.sdfilter = mat(sdfilter, *L, *N, false, true);
	fresult.sdfilter = mat(*L, *N);
	// converts filter time to C++
	//fresult.timefilter = vec(timefilter, *N, false, true);
	fresult.timefilter = vec(*N);

	// final call to filter function
	ukbf(OBSF, DATA, TIME, X0, R0, Q0, PP0, &fresult);

	//double* xfiltert = fresult.xfilter.memptr();
	mat xfiltertest = fresult.xfilter;

	memcpy(xfilter, fresult.xfilter.memptr(), (*L)*(*N) * sizeof(double));
	memcpy(sdfilter, fresult.sdfilter.memptr(), (*L)*(*N) * sizeof(double));
	memcpy(Pfilter, fresult.Pfilter.memptr(), (*L)*(*L)*(*N) * sizeof(double));

	
	
	//Pfilter = fresult.Pfilter.memptr();
	//xfilter = fresult.xfilter.memptr();
	//timefilter = fresult.timefilter.memptr();
	//sdfilter = fresult.sdfilter.memptr();
	
}

//void UTpred_R(double* x, double* PP, double *N, double* time, double* yt, double* yPt, double* yPtc) {
//	// wrapper that takes R inputs and converts them to C++ armadillo inputs for the C++ functions
//	
//	// initial vector of states
//	vec X = vec(x, *N, false, true);
//	// initial covariance
//	mat pp = mat(PP, *N, *N, false, true);
//	// initial time vector 
//	vec TIME = vec(time, 2, false, true);
//	// initial output from UTpred
//	//vec YT = vec(yt, *N, false, true);
//	//mat ypt = mat(yPt, *N, *N, false, true);
//	//mat yptc = mat(yPtc, *N, *N, false, true);
//
//	UTdataOut result;
//	
//
//	UTpred(X, pp, TIME, &result);
//
//}
