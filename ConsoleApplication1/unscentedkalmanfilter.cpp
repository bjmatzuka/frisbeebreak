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
	double *yin = N_VGetArrayPointer(y);
	double *ydotin = N_VGetArrayPointer(ydot);
	yin;
	ydotin;
	//fpointer(&t, (double*) y->content, (double*) ydot->content);
	//fpointer(&t, (double*)(N_VGetArrayPointer(y)), (double*)(N_VGetArrayPointer(ydot)));
	fpointer(&t, yin, ydotin);
	return 0;
}


void predintegrator(double t0, double tf, vec state, ODEout *result) {
	
	// time vector decomposition into T0 and Tfinal   //
	realtype T0;
	realtype Tfinal;
	T0 = t0;
	Tfinal = tf;

	// calculate number of states in system
	double L;
	L = state.n_elem;

	// Prediction Step: 
    // integration of model 

		// number of integration steps
		int Num = 20;

		// maybe not necessary, but here for now   //
		// set up matrix for storing results       //
		mat integrate_store;
		integrate_store.zeros(Num + 1, L + 1);


		// set initial value for int_store
		integrate_store(0, 0) = T0;

		// integrator intialization
		//int Num = 200;
		//realtype T0 = 0;
		//realtype Tfinal = 10;
		//realtype theta0 = 1;
		realtype reltol = 1e-6;
		realtype abstol = 1e-8;
		realtype t;
		int flag, k, j;
		N_Vector y = NULL;
		void* cvode_mem = NULL;
		/* Create serial vector of length NEQ for I.C. */
		y = N_VNew_Serial(L);


		// Set initial conditions for each integration from ensemble
		int jj;
		for (jj = 0; jj < state.n_rows; ++jj) {
			NV_Ith_S(y, jj) = state(jj);
			integrate_store(0, jj + 1) = state(jj);
		}

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

			// store integration results from each iteration
			integrate_store(k, 0) = t;
			for (j = 1; j <= L; ++j) {
				integrate_store(k, j) = NV_Ith_S(y, j - 1);
			}
			// end of results storage loop

		}
		// end of ODE integration

		N_VDestroy_Serial(y); /* Free y vector */
		CVodeFree(&cvode_mem); /* Free integrator memory */

		// store final result of integration
		result->odestate = trans(integrate_store(Num, span(1, L)));
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
	data_orig_new.zeros(Num + 1, n+1);
	mat data_new;
	data_new.zeros(Num + 1, n+1);

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

void sigmapoints_sqrt(mat m, mat P, SigmaOut *result) {

	// set up tuning parameters
	double alpha = 1;
	double beta = 2;
	double kappa = 0;
	

	// initilize unscented transform parameters
	double n;
	double c;
	double np;
	double sc;
	n = m.n_elem;
	np = 2 * n + 1;

	c = alpha*alpha*(n + kappa);
	sc = sqrt(c);

	mat X;
	vec z0 = zeros(n, 1);
	X = repmat(m, 1, np) + sc*(join_horiz(join_horiz(z0, -P), P));

	// Form the weights
	double lambda;
	vec wm1;
	vec wc1;
	vec wn;
	vec wm;
	vec wc;

	lambda = c - n;

	wm1 = lambda / (n + lambda);
	wc1 = wm1 + (1 - alpha*alpha - beta);
	wn = 1 / (2 * (n + lambda));

	wm = trans(join_horiz(wm1, repmat(wn, 1, 2*n)));
	wc = trans(join_horiz(wc1, repmat(wn, 1, 2*n)));

	mat tmp;
	mat W;

	tmp = eye(np, np) - repmat(wm, 1, np);
	W = tmp*diagmat(wc)*trans(tmp);

	result->W = W;
	result->X = X;
	result->wm = wm;
	result->wc = wc;


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
// end of UKBF function

void enkf(mat obsf, mat data, vec time, vec x0, mat R, mat Q, mat P0, FilterOut *result) {
	// Ensemble Kalman Filter for solving SDE
	//
	//			INPUTS:
	//		%			M : operator matrix for observations
	//		%           data : data points used for filter(column vector)
	//		%           time : time period observations occur over
	//		%           x0 : initial condition for system
	//		%           R : Observation noise covariance, constant
	//		%           V : Process noise covariance
	//		%           P0 : initial state filter covariance
	//		%		    q : parameter values
	//		%       OUTPUTS :
	//		%           out.xfilter : state filter output
	//		%           out.P : State covariance matrices for each time
	//		%           out.time : time scale
	//		%           out.data : original data used for filter
	//		%           out.tsd : +/ -3 std.deviations of state filter


		// initilize the variables
	double L;
	double N;
	double Pn;

	Pn = 50;             // number of particles (can be input; hardcode for now)
	L = x0.n_elem;       // number of states
	N = data.n_rows;     // length of observations


	// set up ensemble matrix
	mat Ap = zeros(L, Pn);
	mat A = repmat(x0, 1, Pn) + Q*randn(L, Pn);

	// assign initial values to the respective variables
	result->xfilter.col(0) = x0;
	result->Pfilter.slice(0) = P0;
	result->sdfilter.col(0) = sqrt(diagvec(P0));

	// integrator results initialization
	ODEout odestate;

	// start filter loop
	int i;
	for (i = 1; i < N; ++i) {

		// time vector decomposition into T0 and Tfinal   //
		//realtype T0;
		//realtype Tfinal;
		//T0 = time(i - 1);
		//Tfinal = time(i);

		// Prediction Step: 
		// push each 'particle' of the ensemble through the model
		int ii;
		for (ii = 0; ii < Pn; ++ii) {
			// integration of model for each particle
			predintegrator(time(i - 1), time(i), A.col(ii), &odestate);
			// store results of integration for each particle
			Ap.col(ii) = odestate.odestate;

		}
		// end of particle integration routine


			//// number of integration steps
			//int Num = 20;

			//// maybe not necessary, but here for now   //
			//mat data_orig_new;
			//data_orig_new.zeros(Num + 1, L + 1);
			//mat data_new;
			//data_new.zeros(Num + 1, L + 1);

			//// set initial value for data_orig and data_new
			//data_orig_new(0, 0) = T0;
			//data_new(0, 0) = T0;

			//// integrator intialization
			////int Num = 200;
			////realtype T0 = 0;
			////realtype Tfinal = 10;
			////realtype theta0 = 1;
			//realtype reltol = 1e-6;
			//realtype abstol = 1e-8;
			//realtype t;
			//int flag, k;
			//N_Vector y = NULL;
			//void* cvode_mem = NULL;
			///* Create serial vector of length NEQ for I.C. */
			//y = N_VNew_Serial(L);

			////NV_Ith_S(y, 0) = theta0;
			////NV_Ith_S(y, 1) = 0;

			//// Set initial conditions for each integration from ensemble
			//int jj;
			//for (jj = 0; jj < A.n_rows; ++jj) {
			//	NV_Ith_S(y, jj) = A(jj, ii);
			//	data_orig_new(0, jj + 1) = A(jj, ii);
			//}

			////cout << NV_Ith_S(y, 0) << "IC 1" << endl;
			////cout << NV_Ith_S(y, 1) << "IC 2" << endl;
			////Xi.print("IC Match");

			///* Set up solver */
			//cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
			//if (cvode_mem == 0) {
			//	fprintf(stderr, "Error in CVodeMalloc: could not allocate\n");
			//	//return -1;
			//}
			///* Call CVodeMalloc to initialize the integrator memory */
			//flag = CVodeInit(cvode_mem, f_wrap, T0, y);
			//if (flag < 0) {
			//	fprintf(stderr, "Error in CVodeInit: %d\n", flag);
			//	//return -1;
			//}
			//flag = CVodeSStolerances(cvode_mem, reltol, abstol);
			//if (flag < 0) {
			//	fprintf(stderr, "Error in CVodeSStolerances: %d\n", flag);
			//	//return -1;
			//}
			////flag = CVodeSetStopTime(cvode_mem, Tfinal);
			////if (flag < 0) {
			////	fprintf(stderr, "Error in CVodeSetStopTime: %d\n", flag);
			////}
			////while (t<Tfinal){
			////	flag = CVode(cvode_mem, Tfinal, y, &t, CV_NORMAL);
			////	if (flag < 0) {
			////				fprintf(stderr, "Error in CVode: %d\n", flag);
			////return -1;
			////		}
			////data_orig_new(0, 1) = 1;
			//t = T0;
			//for (k = 1; k <= Num; ++k) {
			//	realtype tout = T0 + (k*(Tfinal - T0) / Num);
			//	//cout << "time out" << tout << endl;
			//	if (CVode(cvode_mem, tout, y, &t, CV_NORMAL) < 0) {
			//		fprintf(stderr, "Error in CVode: %d\n", flag);
			//		//return -1;
			//	}
			//	//printf("%g %.16e %.16e\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1));
			//	//t.print("Time");

			//	data_orig_new(k, 0) = t;
			//	data_orig_new(k, 1) = NV_Ith_S(y, 0);
			//	data_orig_new(k, 2) = NV_Ith_S(y, 1);

			//	//data(k, 0) = t;
			//	//data(k, 1) = data_orig(k, 1) + 0.01*randn(1)[0];
			//	//data(k, 2) = data_orig(k, 2) + 0.01*randn(1)[0];

			//}

			//// end of ODE integration
			//N_VDestroy_Serial(y); /* Free y vector */
			//CVodeFree(&cvode_mem); /* Free integrator memory */
			//					   //data_orig_new.print("check");
			//// store Ensemble integration results
			//Ap.col(ii) = trans(data_orig_new(Num, span(1, L)));

		//}
		// end of particle integration routine

		// Analysis Step: 
		// update the estimate of the state given the obervation
		// ie, calculate posterior through likelihood function

		// calculate ensemble perturbation matrix(51)
		mat I;
		mat Abar;
		mat Aprime;
		I = (1 / Pn)*ones(Pn, Pn);
		Abar = Ap*I;
		Aprime = Ap - Abar;

		// calculate the measurement matrix (54)
		mat D;
		D = repmat(trans(data.row(i)), 1, Pn);

		//calculate measurement perturbation matrix (55)
		mat E;
		E = R*randn(data.n_cols, Pn);
		//E = R*randn(length(data(k, :)'),N);

		// calculate measurement error covariance matrix, C_ee (56)
		mat C_ee;
		C_ee = (1 / (Pn - 1))*E*trans(E);

		// calculate matrix holding measurements of ensemble perturbations
		// and other matrices required for update equation
		// (58, 60, 61, 63)
		mat Dprime;
		mat S;
		mat C;
		mat X;

		Dprime = D - obsf*A;
		S = obsf*Aprime;
		C = S*trans(S) + (Pn - 1)*C_ee;
		X = eye(Pn,Pn) + trans(S)*solve(C, Dprime);

		// Update equation, A^a (62)
		A = Ap*X;

		// find mean and covariance of updated ensemble for filter
		mat meanA;
		mat covA;

		meanA = mean(A, 1);
		covA = (1/(Pn-1))*(A - repmat(meanA,1,Pn))*trans(A - repmat(meanA,1,Pn));

		result->xfilter.col(i) = meanA;
		result->Pfilter.slice(i) = covA;
		result->sdfilter.col(i) = sqrt(diagvec(result->Pfilter.slice(i)));

	}
	// end of filter loop iteration
}
// end of EnKF function


void etkf(mat obsf, mat data, vec time, vec x0, mat R, mat Q, mat P0, FilterOut *result) {
	// Ensemble Transform Kalman Filter for solving SDE
	//
	//			INPUTS:
	//		%			M : operator matrix for observations
	//		%           data : data points used for filter(column vector)
	//		%           time : time period observations occur over
	//		%           x0 : initial condition for system
	//		%           R : Observation noise covariance, constant
	//		%           V : Process noise covariance
	//		%           P0 : initial state filter covariance
	//		%		    q : parameter values
	//		%       OUTPUTS :
	//		%           out.xfilter : state filter output
	//		%           out.P : State covariance matrices for each time
	//		%           out.time : time scale
	//		%           out.data : original data used for filter
	//		%           out.tsd : +/ -3 std.deviations of state filter


	// initilize the variables
	double L;
	double N;
	double Pn;
	double r;
	
	r = 0.05;            // inflation factor
	Pn = 50;             // number of particles (can be input; hardcode for now)
	L = x0.n_elem;       // number of states
	N = data.n_rows;     // length of observations


	// set up ensemble matrix
	mat Ap = zeros(L, Pn);
	mat A = repmat(x0, 1, Pn) + sqrt(P0)*randn(L, Pn);

	// assign initial values to the respective variables
	result->xfilter.col(0) = x0;
	result->Pfilter.slice(0) = P0;
	result->sdfilter.col(0) = sqrt(diagvec(P0));

	// initialize integration results
	ODEout odestate;

	mat invR;
	invR = inv(R);
	// start filter loop
	int i;
	for (i = 1; i < N; ++i) {

		// time vector decomposition into T0 and Tfinal   //
		//realtype T0;
		//realtype Tfinal;
		//T0 = time(i - 1);
		//Tfinal = time(i);

		// Prediction Step: 
		// push each 'particle' of the ensemble through the model
		int ii;
		for (ii = 0; ii < Pn; ++ii) {

			//integration of model with each particle 
			predintegrator(time(i - 1), time(i), A.col(ii), &odestate);

			// store integration results for each particle
			Ap.col(ii) = odestate.odestate;

		}
		// end of integration routine for each sigma


		//	// number of integration steps
		//	int Num = 20;

		//	// maybe not necessary, but here for now   //
		//	mat data_orig_new;
		//	data_orig_new.zeros(Num + 1, L + 1);
		//	mat data_new;
		//	data_new.zeros(Num + 1, L + 1);

		//	// set initial value for data_orig and data_new
		//	data_orig_new(0, 0) = T0;
		//	data_new(0, 0) = T0;

		//	// integrator intialization
		//	//int Num = 200;
		//	//realtype T0 = 0;
		//	//realtype Tfinal = 10;
		//	//realtype theta0 = 1;
		//	realtype reltol = 1e-6;
		//	realtype abstol = 1e-8;
		//	realtype t;
		//	int flag, k;
		//	N_Vector y = NULL;
		//	void* cvode_mem = NULL;
		//	/* Create serial vector of length NEQ for I.C. */
		//	y = N_VNew_Serial(L);

		//	//NV_Ith_S(y, 0) = theta0;
		//	//NV_Ith_S(y, 1) = 0;

		//	// Set initial conditions for each integration from ensemble
		//	int jj;
		//	for (jj = 0; jj < A.n_rows; ++jj) {
		//		NV_Ith_S(y, jj) = A(jj, ii);
		//		data_orig_new(0, jj + 1) = A(jj, ii);
		//	}

		//	//cout << NV_Ith_S(y, 0) << "IC 1" << endl;
		//	//cout << NV_Ith_S(y, 1) << "IC 2" << endl;
		//	//Xi.print("IC Match");

		//	/* Set up solver */
		//	cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
		//	if (cvode_mem == 0) {
		//		fprintf(stderr, "Error in CVodeMalloc: could not allocate\n");
		//		//return -1;
		//	}
		//	/* Call CVodeMalloc to initialize the integrator memory */
		//	flag = CVodeInit(cvode_mem, f_wrap, T0, y);
		//	if (flag < 0) {
		//		fprintf(stderr, "Error in CVodeInit: %d\n", flag);
		//		//return -1;
		//	}
		//	flag = CVodeSStolerances(cvode_mem, reltol, abstol);
		//	if (flag < 0) {
		//		fprintf(stderr, "Error in CVodeSStolerances: %d\n", flag);
		//		//return -1;
		//	}
		//	//flag = CVodeSetStopTime(cvode_mem, Tfinal);
		//	//if (flag < 0) {
		//	//	fprintf(stderr, "Error in CVodeSetStopTime: %d\n", flag);
		//	//}
		//	//while (t<Tfinal){
		//	//	flag = CVode(cvode_mem, Tfinal, y, &t, CV_NORMAL);
		//	//	if (flag < 0) {
		//	//				fprintf(stderr, "Error in CVode: %d\n", flag);
		//	//return -1;
		//	//		}
		//	//data_orig_new(0, 1) = 1;
		//	t = T0;
		//	for (k = 1; k <= Num; ++k) {
		//		realtype tout = T0 + (k*(Tfinal - T0) / Num);
		//		//cout << "time out" << tout << endl;
		//		if (CVode(cvode_mem, tout, y, &t, CV_NORMAL) < 0) {
		//			fprintf(stderr, "Error in CVode: %d\n", flag);
		//			//return -1;
		//		}
		//		//printf("%g %.16e %.16e\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1));
		//		//t.print("Time");

		//		data_orig_new(k, 0) = t;
		//		data_orig_new(k, 1) = NV_Ith_S(y, 0);
		//		data_orig_new(k, 2) = NV_Ith_S(y, 1);

		//		//data(k, 0) = t;
		//		//data(k, 1) = data_orig(k, 1) + 0.01*randn(1)[0];
		//		//data(k, 2) = data_orig(k, 2) + 0.01*randn(1)[0];

		//	}

		//	// end of ODE integration
		//	N_VDestroy_Serial(y); /* Free y vector */
		//	CVodeFree(&cvode_mem); /* Free integrator memory */
		//						   //data_orig_new.print("check");
		//						   // store Ensemble integration results
		//	Ap.col(ii) = trans(data_orig_new(Num, span(1, L)));

		//}
		// end of particle integration routine

		Ap = Ap + sqrt(Q)*randn(L, Pn);

		// Analysis Step: 
		// update the estimate of the state given the obervation
		// ie, calculate posterior through likelihood function

		// G, the ensemble obs matrix
		mat G = zeros(data.n_cols,Pn);
		double kk;
		for (kk = 0; kk < Pn; ++kk) {
			G.col(kk) = obsf*Ap.col(kk);
		}

		// calculate, Vobs, (nonlinear) obs matrix
		mat I2;
		mat Vbar;
		mat Vobs;
		I2 = (1 / Pn)*ones(Pn, Pn);
		Vbar = G*I2;
		Vobs = G - Vbar;

		// calculate ensemble perturbation matrix(51)
		mat I;
		mat Abar;
		mat Aprime;
		I = (1 / Pn)*ones(Pn, Pn);
		Abar = Ap*I;
		Aprime = Ap - Abar;

		// variance inflation step
		Aprime = sqrt(1 + r)*Aprime;
		Vobs = sqrt(1 + r)*Vobs;

		// calculate SVD of J
		mat J;
		mat U;
		vec s;
		mat V;

		J = ((Pn - 1) / (1 + r))*eye(Pn, Pn) + trans(Vobs)*invR*Vobs;
		svd_econ(U, s, V, J);

		// Kalman Gain
		mat K;
		K = Aprime*pinv(J)*trans(Vobs)*invR;

		// update equations
		// posterior mean, u_a
		mat u_a;
		u_a = Abar.col(0) + K*(trans(data.row(i)) - obsf*Abar.col(0));

		// compute transformation matrix, T
		mat T;
		mat smat = diagmat(s);
		T = sqrt(Pn - 1)*U*chol(inv(smat))*trans(U);

		// posterior perturbation matrix of ensembles
		mat Um;
		Um = Aprime*T;
		
		// posterior ensembles
		mat u_mean;
		u_mean = repmat(u_a, 1, Pn);
		A = u_mean + Um;

		// find mean and covariance of updated ensemble for filter
		mat meanA;
		mat covA;

		meanA = mean(A, 1);
		covA = (1 / (Pn - 1))*(A - repmat(meanA, 1, Pn))*trans(A - repmat(meanA, 1, Pn));

		result->xfilter.col(i) = meanA;
		result->Pfilter.slice(i) = covA;
		result->sdfilter.col(i) = sqrt(diagvec(result->Pfilter.slice(i)));

	}
	// end of filter loop iteration; each iteration is a data point
}
// end of ETKF function


void srukf(mat obsf, mat data, vec time, vec x0, mat R, mat Q, mat P0, FilterOut *result) {
//	%Sqrt UKBF : Square Root Unscented Kalman Filter
//		%
//		%   This implements a continuous - discrete square root unscented kalman bucy filter
//		% to estimate the state of a nonlinear observation function.
//		%
//		%      INPUTS :
//		%           dynfun : rhs of ODE system(model) from which parameters are
//		%                   being estimated(column vector)
//		%			obsfun : general function for observations(d = G(x, w))
//		%			data : data points used for filter(column vector)
//		%			time : time period observations occur over
//		%           x0 : initial guess of parameter value
//		%           R : Measurement noise covariance, constant
//		%           V : Process noise covariance, constant
//		%           Px0 : initial covariance of parameter values
//		%           q : parameter values
//		%       OUTPUTS :
//		%           out.xfilter : filter of parameter estimates(last value is final
//		%                        estimate of parameter; best fit)
//		%			out.Px : covariance matrix at each filter point
//		%           out.time : time scale
//		%           out.data : original data used for estimation
//		%           out.tsd : three standard deviations for estimates
//		%
//		% Brett Matzuka, qP, November 2018
//		%
//		% implementation was done using 'Kalman Filtering and Neural Networks'
//		% edited by Haykin, chapter 7
	
// initilize the variales
	double L;
	double N;

	L = x0.n_elem;
	N = data.n_rows;


	// assign initial values to the respective variables
	result->xfilter.col(0) = x0;
	result->Pfilter.slice(0) = P0;
	result->sdfilter.col(0) = sqrt(diagvec(P0));
	
	//correct initial covariances
	R = chol(R);
	Q = chol(Q);

	// set up inputs
	vec xminus;
	mat Pminus;
    // initialize storage of sigma point and ode info
	SigmaOut sigmaparms;
	ODEout odestate;

	// main filter loop

	// start filter loop
	int i;
	for (i = 1; i < N; ++i) {
		// time update of state and covariance
		xminus = result->xfilter.col(i - 1);
		Pminus = result->Pfilter.slice(i - 1);

		// create sigma point stencil for model
		sigmapoints_sqrt(xminus, Pminus, &sigmaparms);

		mat Xhat = zeros(L, 2 * L + 1);
		// Integration loop: passing sigma point stencil through the dynamic model and propagating
		// forward
		//
		// Prediction Step: 
		// push each sigma point of the stencil through the model
		int ii;
		for (ii = 0; ii < sigmaparms.X.n_cols; ++ii) {

			//integration of model for each sigma point 
			mat stateinit;
			stateinit = sigmaparms.X;
			predintegrator(time(i - 1), time(i), stateinit.col(ii), &odestate);

			// store integration results for each particle
			Xhat.col(ii) = odestate.odestate;

		}
		// end of particle integration routine

		//int Num;
		//Num = 20;

		//// time vector decomposition into T0 and Tfinal   //
		//realtype T0;
		//realtype Tfinal;
		//T0 = time(i-1);
		//Tfinal = time(i);

		//// maybe not necessary, but here for now   //
		//mat data_orig_new;
		//data_orig_new.zeros(Num + 1, L + 1);
		//mat data_new;
		//data_new.zeros(Num + 1, L + 1);

		//// set initial value for data_orig and data_new
		//data_orig_new(0, 0) = T0;
		//data_new(0, 0) = T0;

		//// for loop over the sigma points
		//int ii;
		//for (ii = 0; ii < sigmaparms.X.n_cols; ++ii) {

		//	

		//	// integrator intialization
		//	//int Num = 200;
		//	//realtype T0 = 0;
		//	//realtype Tfinal = 10;
		//	//realtype theta0 = 1;
		//	realtype reltol = 1e-6;
		//	realtype abstol = 1e-8;
		//	realtype t;
		//	int flag, k;
		//	N_Vector y = NULL;
		//	void* cvode_mem = NULL;
		//	/* Create serial vector of length NEQ for I.C. */
		//	y = N_VNew_Serial(L);

		//	//NV_Ith_S(y, 0) = theta0;
		//	//NV_Ith_S(y, 1) = 0;
		//	int jj;
		//	for (jj = 0; jj < sigmaparms.X.n_rows; ++jj) {
		//		NV_Ith_S(y, jj) = sigmaparms.X(jj, ii);
		//		data_orig_new(0, jj + 1) = sigmaparms.X(jj, ii);
		//	}
		//	//cout << NV_Ith_S(y, 0) << "IC 1" << endl;
		//	//cout << NV_Ith_S(y, 1) << "IC 2" << endl;
		//	//Xi.print("IC Match");

		//	/* Set up solver */
		//	cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
		//	if (cvode_mem == 0) {
		//		fprintf(stderr, "Error in CVodeMalloc: could not allocate\n");
		//		//return -1;
		//	}
		//	/* Call CVodeMalloc to initialize the integrator memory */
		//	flag = CVodeInit(cvode_mem, f_wrap, T0, y);
		//	if (flag < 0) {
		//		fprintf(stderr, "Error in CVodeInit: %d\n", flag);
		//		//return -1;
		//	}
		//	flag = CVodeSStolerances(cvode_mem, reltol, abstol);
		//	if (flag < 0) {
		//		fprintf(stderr, "Error in CVodeSStolerances: %d\n", flag);
		//		//return -1;
		//	}
		//	//flag = CVodeSetStopTime(cvode_mem, Tfinal);
		//	//if (flag < 0) {
		//	//	fprintf(stderr, "Error in CVodeSetStopTime: %d\n", flag);
		//	//}
		//	//while (t<Tfinal){
		//	//	flag = CVode(cvode_mem, Tfinal, y, &t, CV_NORMAL);
		//	//	if (flag < 0) {
		//	//				fprintf(stderr, "Error in CVode: %d\n", flag);
		//	//return -1;
		//	//		}
		//	//data_orig_new(0, 1) = 1;
		//	t = T0;
		//	for (k = 1; k <= Num; ++k) {
		//		realtype tout = T0 + (k*(Tfinal - T0) / Num);
		//		//cout << "time out" << tout << endl;
		//		if (CVode(cvode_mem, tout, y, &t, CV_NORMAL) < 0) {
		//			fprintf(stderr, "Error in CVode: %d\n", flag);
		//			//return -1;
		//		}
		//		//printf("%g %.16e %.16e\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1));
		//		//t.print("Time");

		//		data_orig_new(k, 0) = t;
		//		data_orig_new(k, 1) = NV_Ith_S(y, 0);
		//		data_orig_new(k, 2) = NV_Ith_S(y, 1);

		//		//data(k, 0) = t;
		//		//data(k, 1) = data_orig(k, 1) + 0.01*randn(1)[0];
		//		//data(k, 2) = data_orig(k, 2) + 0.01*randn(1)[0];

		//	}
		//	// end of ODE integration

		//	N_VDestroy_Serial(y); /* Free y vector */
		//	CVodeFree(&cvode_mem); /* Free integrator memory */
		//						   //data_orig_new.print("check");

		//						   // store state at final integrator time
		//	Xhat.col(ii) = trans(data_orig_new(Num, span(1, L)));
		//}
		// for loop for integration occurs here!!!
		//
		//
		//end of integration loop

		// generate the prior estimate for the state, weighted mean. (7.54)
		mat xhat_;
		xhat_ = Xhat*sigmaparms.wm;

		// measurement update of covariance
		//   this is where square root ukf differs from ukf
		// --------------------------------------------------
		mat Xstar;
		Xstar = Xhat - repmat(xhat_, 1, 2 * L + 1);
		double sqrt_wc;
		sqrt_wc = sqrt(sigmaparms.wc(1));

		// qr factorization
		mat waste;
		mat Sk_;
		qr_econ(waste, Sk_, trans(join_horiz(sqrt_wc*Xstar.cols(1, 2 * L + 1), Q)));

		// cholesky update to complete the covariance prediction
		if (sigmaparms.wc(0) > 0) {
			Sk_ = chol(Sk_*trans(Sk_) + (sqrt(sqrt(abs(sigmaparms.wc(0))))*Xstar.col(0))*trans(sqrt(sqrt(abs(sigmaparms.wc(0))))*Xstar.col(0)), "lower");
		}
		else {
			Sk_ = chol(Sk_*trans(Sk_) - (sqrt(sqrt(abs(sigmaparms.wc(0))))*Xstar.col(0))*trans(sqrt(sqrt(abs(sigmaparms.wc(0))))*Xstar.col(0)), "lower");
		}

		// making sigma points for observations
		sigmapoints_sqrt(xhat_, Sk_, &sigmaparms);

		mat Xminus;
		Xminus = sigmaparms.X;

		// push the stencil through the observation function
		mat Y = zeros(size(data,2), 2 * L + 1);
		int jj;
		for (jj = 0; jj < 2 + L + 1; ++jj) {
			Y.col(jj) = obsf*Xminus.col(jj);
		}

		// prior estimate for observation
		mat yhat_;
		yhat_ = Y*sigmaparms.wm;

		// measurement update of covariance
		//   this is where square root ukf differs from ukf
		// --------------------------------------------------
		mat Ystar;
		Ystar = Y - repmat(yhat_, 1, 2 * L + 1);

		// qr factorization
		mat waste2;
		mat Syk;
		qr_econ(waste2, Syk, trans(join_horiz(sqrt_wc*Ystar.cols(1, 2 * L + 1), R)));

		// cholesky update to complete the covariance prediction
		if (sigmaparms.wc(0) > 0) {
			Syk = chol(Syk*trans(Syk) + (sqrt(sqrt(abs(sigmaparms.wc(0))))*Ystar.col(0))*trans(sqrt(sqrt(abs(sigmaparms.wc(0))))*Ystar.col(0)), "lower");
		}
		else {
			Syk = chol(Syk*trans(Syk) - (sqrt(sqrt(abs(sigmaparms.wc(0))))*Ystar.col(0))*trans(sqrt(sqrt(abs(sigmaparms.wc(0))))*Ystar.col(0)), "lower");
		}


		// obsevation cross covariance
		mat Pxy;
		Pxy = Xminus*sigmaparms.W*trans(Y);

		// Gain matrix
		mat Kk;
		Kk = solve(solve(Pxy, trans(Syk)), Syk);

		// measurement update of state filter
		vec xplus;
		xplus = xhat_ + Kk*(trans(data.row(i)) - yhat_);

		// correct covariance: U
		mat U;
		U = Kk*Syk;

		// measurement update of covariance parameter
		Sk_ = trans(Sk_);
		int kk;
		for (kk = 0; kk < size(data, 2); ++kk) {
			Sk_ = chol(Sk_*trans(Sk_) - U.col(kk)*trans(U.col(kk)), "lower");
		}

		// storing covariance
		mat Swplus;
		Swplus = Sk_;

		// store filter and covariances
		result->xfilter.col(i) = xplus;
		result->Pfilter.slice(i) = Swplus*trans(Swplus);
		result->sdfilter.col(i) = sqrt(diagvec(Swplus*trans(Swplus)));

	}
	// end of main filter loop; looping over observations, doing main filter iterations

// end of srukf function
}
// end of srukf function


void ekbf(mat obsf, mat data, vec time, vec x0, mat R, mat Q, mat P0, FilterOut *result) {
	
	// Brett Matzuka, qP, Nov. 2018
	//
	//
	//	EKBF Extended Kalman Bucy Filter
	//	%
	//	%  this implements a continuous - discrete extended kalman filter
	//	%  using Automatic Differentiation to calculate the jacobian
	//	%
	//	%     INPUTS:
	//          model : rhs of ODE system(include values for parameters)
	//	%		obser : general function for observations(z = h(x))
	//	%		data : data points used for filter(column vectors)
	//	%		time : time period observations occur over
	//	%		   R : covariance noise for data, constant
	//	%          Q : covariance noise for process, constant
	//	%          x0 : initial condition for model
	//	%          P0 : initial condition for covariance
	//	%          q : parameter values
	//	%          fdad : describes how to calculate jacobian
	//	%               fdad = 0 (default); finite difference
	//	%               fdad = 1; Automatic differentiation jacobian
	//	%          options: ode solver options
	//	%     OUTPUTS :
	//	%          out.xfilter : filter output
	//	%          out.time : time scale
	//	%          out.P : covariance matrices for each time
	//	%          out.sd : +/ -standard deviations


	// initilize the variales
	double L;
	double N;

	L = x0.n_elem;
	N = data.n_rows;


	// assign initial values to the respective variables
	result->xfilter.col(0) = x0;
	result->Pfilter.slice(0) = P0;
	result->sdfilter.col(0) = sqrt(diagvec(P0));


	// start filter loop
	int i;
	for (i = 1; i < N; ++i) {

		// PREDICTION STEP:  integration of model state mean
		// 
		//
		// integration goes here!
		//
		//
		// end of integration loop

		// integration of model state covariance
		//
		//
		// integration of covariance goes here!
		// 
		//
		// end of covariance integration loop



		// CORRECTION STEP
		mat K;


	}
	// end of filter loop
}
// end of EKBF code

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
	double* xfilter, double* timefilter, double* Pfilter, double* sdfilter, 
	int (*rfuncinput)(double* , double*, double*)) {
	
	// wrapper that takes R inputs and converts them to C++ armadillo inputs for the C++ functions

	// understanding of pointer dimensions
	// N = length of time/data
	// L = no. of states
	// M = no. of observed states (M<=L)

	//fpointer = &f_rtest;
	fpointer = rfuncinput;

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

void etkf_R(double* obsf, double* data, long* M, double* time, long* N, double* x0, long* L,
	double* R, double* Q, double* P0,
	double* xfilter, double* timefilter, double* Pfilter, double* sdfilter,
	int (*rfuncinput)(double*, double*, double*)) {

	// wrapper that takes R inputs and converts them to C++ armadillo inputs for the C++ functions

	// understanding of pointer dimensions
	// N = length of time/data
	// L = no. of states
	// M = no. of observed states (M<=L)

	//fpointer = &f_rtest;
	fpointer = rfuncinput;

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
	fresult.Pfilter = cube(*L, *L, *N);
	// converts filter means to C++
	//fresult.xfilter = mat(xfilter, *L, *N, false, true);
	fresult.xfilter = mat(*L, *N);
	// converts filter standard deviation to C++
	//fresult.sdfilter = mat(sdfilter, *L, *N, false, true);
	fresult.sdfilter = mat(*L, *N);
	// converts filter time to C++
	//fresult.timefilter = vec(timefilter, *N, false, true);
	fresult.timefilter = vec(*N);

	// final call to filter function
	etkf(OBSF, DATA, TIME, X0, R0, Q0, PP0, &fresult);

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
