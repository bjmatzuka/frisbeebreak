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
#include "UKF.h"

using namespace std;
using namespace arma;

int ODEtest1();

static int f(realtype t, N_Vector y, N_Vector ydot, void *f_data)
{
	realtype theta = NV_Ith_S(y, 0);
	realtype omega = NV_Ith_S(y, 1);
	realtype omegap = -sin(theta);
	NV_Ith_S(ydot, 0) = omega;
	NV_Ith_S(ydot, 1) = omegap;
	return 0;
}

static int f2(realtype t, N_Vector y, N_Vector ydot, void *f_data)
{
	realtype theta = NV_Ith_S(y, 0);
	realtype omega = NV_Ith_S(y, 1);
	realtype kparm = NV_Ith_S(y, 2);
	realtype omegap = -sin(theta);
	NV_Ith_S(ydot, 0) = omega;
	NV_Ith_S(ydot, 1) = kparm*omegap;
	NV_Ith_S(ydot, 2) = 0;
	return 0;
}


// this needs to have R function inputs
int(*fpointer2)(double*, double*, double*);

static int forig(realtype t, N_Vector y, N_Vector ydot, void *f_data)
{
	realtype theta = NV_Ith_S(y, 0);
	realtype omega = NV_Ith_S(y, 1);
	realtype omegap = -sin(theta);
	NV_Ith_S(ydot, 0) = omega;
	NV_Ith_S(ydot, 1) = omegap;
	return 0;
}

static int f_rtest2(double* t, double* y, double* ydot)
{
	ydot[0] = y[1];
	ydot[1] = -sin(y[0]);
	return 0;
}


static int f_wrap2(realtype t, N_Vector y, N_Vector ydot, void *f_data)
{
	// this will unpack the CVode inputs and convert them to Cpp.

	//fpointer2(&t, (double*)(y->content), (double*)(ydot->content));
	fpointer2(&t, (double*)(N_VGetArrayPointer(y)), (double*)(N_VGetArrayPointer(ydot)));
	return 0;
}





int testnew1() {
	// basic test 
	//double T = 3.14;
	//cout << T;

	// armadillo test
	mat A(2, 3);
	A(0, 1) = 456.0;  // directly access an element (indexing starts at 0)
	A.print("A:");

	// observation error covariance
	mat R = 0.00001*eye<mat>(2, 2);
	R.print("R:");

	// system error covariance
	mat Q = 1*eye<mat>(2, 2);
	Q.print("Q:");

	// initial conditions for state vector
	vec ic;
	ic = ones(Q.n_rows, 1);
	ic.print("ic:");

	// data file
	mat datatest;
	datatest.zeros(2, 20);
	datatest.print("data:");

	// Square Root Unscented Transform
	// sigma point creation for Square Root Unscented Filter
	// unscented tuning parameters
	//
	// alpha can be 1<= alpha <= 0.0001
	// kappa can be 0 or 3 - state dimension (n)
	// beta is set to be 2 for gaussian
	//mat a(1, 1);
	//mat b(1, 1);
	//mat k(1, 1);
	//a(0, 0) = 1;
	//b(0, 0) = 2;
	//k(0, 0) = 0;
	fpointer2 = &f_rtest2;

	double a;
	double b;
	double k;
	a = 1;
	b = 2;
	k = 0;

	cout << "a:" << a << endl;
	cout << "b:" << b << endl;
	cout << "k:" << k << endl;
	//a.print("a:");
	//b.print("b:");
	//k.print("k:");

	// start of function calculations; m and P will need to be inputs
	mat m(5, 1);
	mat P(5, 5);
	m = randn(5, 1);
	P = ones(5, 5);
	m.print("m:");
	P.print("P:");

	// initiation of input info
	//mat nn(1, 1);
	double nn;
	//mat np(1, 1);
	double np;
	nn = as_scalar(m.n_rows);
	//nn(0, 0) = m.n_rows;
	np = 2 * nn + 1;
	//np(0, 0) = 2 * as_scalar(nn) + 1;
	//nn.print("nn:");
	cout << "nn:" << nn << endl;
	//np.print("np:");
	cout << "np:" << np << endl;
	// weighting parameters
	//mat c(1, 1);
	//mat sc(1, 1);
	double c;
	double sc;
	c = a*a*(nn + k);
	//c(0, 0) = as_scalar(a)*as_scalar(a)*(as_scalar(nn) + as_scalar(k));
	//sc(0, 0) = sqrt(as_scalar(c));
	sc = sqrt(c);

	cout << "c:" << c << endl;
	cout << "sc:" << sc << endl;
	//c.print("c:");
	//sc.print("sc:");

	// sigma points
	mat X(m.n_rows, np + 1 + 2 * as_scalar(P.n_rows));

	X.print("X:");
	// size of X (checking)
	cout << "number of rows: " << X.n_rows << endl;
	cout << "number of columns: " << X.n_cols << endl;

	// Form the weights
	mat L(1, 1);
	
	L(0, 0) = c - as_scalar(nn);
	L.print("L:");

	mat wm(1, 2 * nn + 1);
	wm.fill(1 / (2 * (nn + as_scalar(L))));
	wm(0, 0) = L(0, 0) / (nn + as_scalar(L));
	//wmcols = wm.n_cols;
	//wm.cols(2, wmcols) = 1 / (2 * (as_scalar(nn) + as_scalar(L)));
	wm.print("wm:");

	mat wc(1, 2 * nn + 1);
	wc.fill(1 / (2 * (nn + as_scalar(L))));
	wc(0, 0) = L(0, 0) / (nn + as_scalar(L)) + (1 - a*a + b);

	mat tmpeye = eye<mat>(np, np);
	mat tmpsub(wm.n_cols, np);
	tmpsub.each_col() = trans(wm);
	mat tmp = tmpeye - tmpsub;

	mat W;
	mat wdiag;
	wdiag = diagmat(wc);
	W = tmp*wdiag*trans(tmp);
	//cout << "wm transpose: " << trans(wm) << endl;

	tmpeye.print("tmpeye:");
	tmpsub.print("tmpsub:");
	tmp.print("tmp:");
	W.print("W:");
	
	// Regular Unscented transformation function
	// need to make a function here for this
	//
	
	// inputs: function, state/x, covariance/P, time
	mat x = ones<mat>(2, 1);
	mat PP = randn<mat>(2, 2);
	x(1, 0) = 0;
	PP = 0.1*diagmat(ones(2));

	// unscented tuning parameters: a, b, and k
	//
	// alpha can be 1<= alpha <= 0.0001
	// kappa can be 0 or 3 - state dimension (n)
	// beta is set to be 2 for gaussian
	mat aa(1,1);
	mat bb(1,1);
	mat kk(1,1);
	aa(0, 0) = 1;
	bb(0, 0) = 2;
	kk(0, 0) = 0;

	mat n;
	mat lambda;
	n = x.n_elem;
	lambda = as_scalar(a)*as_scalar(a)*(as_scalar(n) + as_scalar(kk)) - as_scalar(n);

	lambda.print("lambda:");

	cx_mat eigvecP;
	cx_vec eigvalP;
	eig_gen(eigvalP, eigvecP, PP);
	mat cc = zeros<mat>(PP.n_rows, PP.n_cols);
	if (any(eigvalP) < 0)
	{
		cc = real(sqrtmat(PP));
	} 
	else {
		//mat U;
		//vec s;
		//mat V;
		//svd(U, s, V, PP);
		//mat sqrtD = sqrt(diagmat(s));
		//cc = (U*sqrtD)*trans(U);
		cc = trans(chol(PP));
	}
	//eigvalP.print("eigs_val:");
	//eigvecP.print("eigs_vec:");
	cc.print("C matrix:");

	// paran: = sqrt(n+lambda)
	mat den(1, 1);
	mat paran(1, 1);
	mat rootP(cc.n_rows, cc.n_cols);
	den = as_scalar(n) + as_scalar(lambda);
	paran = sqrt(as_scalar(den));
	rootP = cc*as_scalar(paran);
	rootP.print("root P:");

	// calculates the sigma points, Xi = xhat +/- paran*c
	mat Y = repmat(x, 1, as_scalar(n));
	mat XiP = join_horiz(Y - rootP, Y + rootP);
	mat Xi = join_horiz(x, XiP);

	Xi.print("Xi:");

	mat Wm(1, 2 * as_scalar(n) + 1);
	mat Wc(1, 2 * as_scalar(n) + 1);
	Wm.fill(1 / (2 * (as_scalar(n) + as_scalar(lambda))));
	Wc.fill(1 / (2 * (as_scalar(n) + as_scalar(lambda))));
	Wm(0, 0) = as_scalar(lambda) / (as_scalar(n) + as_scalar(lambda));
	Wc(0, 0) = as_scalar(lambda) / (as_scalar(n) + as_scalar(lambda)) + (1 - as_scalar(aa)*as_scalar(aa) + as_scalar(bb));

	// Weight matrix for transformed covariance calcs
	mat WW;
	mat WmMat = repmat(trans(Wm), 1, Wm.n_elem);
	mat eyeL = eye<mat>(2 * as_scalar(n) + 1, 2 * as_scalar(n) + 1);
	WW = (eyeL - WmMat)*diagmat(Wc)*trans(eyeL - WmMat);
	WmMat.print("Wm matrix:");
	WW.print("W matrix:");
	
	//mat odesol;
	//odesol = ODEtest1();
	//odesol.print("ode solution:");

	mat data_orig_new;
	data_orig_new.zeros(200, 3);
	mat data_new;
	data_new.zeros(200, 3);

	// ODE integrator code testing //
	//
	//
	/////////////////////////////////

	// integrator results storage
	mat Yi = zeros(as_scalar(n), Xi.n_cols);
	// mean prediction
	mat xt = zeros(as_scalar(n), 1);
	// covariance, Pt
	mat Pt = zeros(as_scalar(n));
	mat Ptc = Pt;

	// for loop over the sigma points
	int ii;
	for (ii = 0; ii < Xi.n_cols; ++ii) {

		// integrator intialization
		int Num = 200;
		realtype T0 = 0;
		realtype Tfinal = 10;
		realtype theta0 = 1;
		realtype reltol = 1e-4;
		realtype abstol = 1e-5;
		realtype t;
		int flag, ki;
		N_Vector y = NULL;
		void* cvode_mem = NULL;
		/* Create serial vector of length NEQ for I.C. */
		y = N_VNew_Serial(2);

		//NV_Ith_S(y, 0) = theta0;
		//NV_Ith_S(y, 1) = 0;
		int jj;
		for(jj = 0; jj < Xi.n_rows; ++jj) {
			NV_Ith_S(y, jj) = Xi(jj, ii);
			data_orig_new(0, jj + 1) = Xi(jj, ii);
			data_new(0, jj + 1) = Xi(jj, ii);
		}
		/* Set up solver */
		cvode_mem = CVodeCreate(CV_ADAMS, CV_FUNCTIONAL);
		if (cvode_mem == 0) {
			fprintf(stderr, "Error in CVodeMalloc: could not allocate\n");
			return -1;
		}
		/* Call CVodeMalloc to initialize the integrator memory */
		flag = CVodeInit(cvode_mem, f, T0, y);
		if (flag < 0) {
			fprintf(stderr, "Error in CVodeInit: %d\n", flag);
			return -1;
		}
		flag = CVodeSStolerances(cvode_mem, reltol, abstol);
		if (flag < 0) {
			fprintf(stderr, "Error in CVodeSStolerances: %d\n", flag);
			return -1;
		}
		//data_orig_new(0, 1) = 1;
		for (ki = 1; ki < Num; ++ki) {
			realtype tout = ki*Tfinal / Num;
			if (CVode(cvode_mem, tout, y, &t, CV_NORMAL) < 0) {
				fprintf(stderr, "Error in CVode: %d\n", flag);
				return -1;
			}
			//printf("%g %.16e %.16e\n", t, NV_Ith_S(y, 0), NV_Ith_S(y, 1));
			//t.print("Time");
			data_orig_new(ki, 0) = t;
			data_orig_new(ki, 1) = NV_Ith_S(y, 0);
			data_orig_new(ki, 2) = NV_Ith_S(y, 1);


			data_new(ki, 0) = t;
			data_new(ki, 1) = data_orig_new(ki, 1) + 0.1*randn(1)[0];
			data_new(ki, 2) = data_orig_new(ki, 2) + 0.1*randn(1)[0];
		}
		// end of ODE integration

		// store state at final integrator time
		Yi.col(ii) = trans(data_orig_new(Num-1,span(1,as_scalar(n))));
		// calculate mean prediction
		xt = xt + Yi.col(ii)*Wm(ii);

	}
	// end of sigma point loop
	data_orig_new.print("output:");
	Yi.print("Yi:");
	xt.print("xt:");
	//covariance and cross covariance calculation
	Pt = Yi*WW*trans(Yi);
	Ptc = Xi*WW*trans(Yi);

	Pt.print("covariance:");
	Ptc.print("cross covariance:");

	// Sigma point creation and prediction 
	vec x0 = ones(2,1);
	x0(1) = 0.4472;
	vec time = zeros(2, 1);
	time(1) = 10;
	UTdataOut result0;
	UTpred(x0, PP, time, &result0);

	result0.yt.print("pred transformed mean:");
	result0.yPt.print("pred transformed covariance:");

	mat Pkp = zeros<mat>(as_scalar(n), as_scalar(n));
	Pkp = Pt + Q;

	// sigma point creation for observsation function
	//
	// assuming observing linear function of states
	mat obsfunc = eye<mat>(as_scalar(n),as_scalar(n));
	UTdataOut result;
	UTdata(obsfunc, xt, Pkp, xt, &result);

	result.yt.print("transformed mean:");
	result.yPt.print("transformed covariance:");

	mat Sk = zeros(as_scalar(n), as_scalar(n));
	Sk = result.yPt + R;

	// data for testing
	double Num;
	Num = 200;
	mat datat = ones(2,1);
	datat = trans(data_orig_new(Num - 1, span(1, as_scalar(n))));
	datat.print("data check:");

	// calculate the update step (correction) for the filter
	mat LL;
	mat U;
	mat U2;
	LL = chol(Sk);
	U = solve(LL,result.yPtc);
	U2 = trans(solve(trans(LL), trans(result.yPtc)));
	U.print("U:");
	U2.print("U2:");

	vec xc = zeros(x0.n_rows,1);
	mat PC = zeros(as_scalar(n), as_scalar(n));
	mat adjust;
	PC = Pkp - U*trans(U);
	adjust = U*(solve(trans(LL), datat-result.yt));
	xc = result0.yt + U*(solve(trans(LL), datat - result.yt));
	xc.print("xc 1:");
	PC.print("PC 1:");
	adjust.print("adjustment");

	vec xc2 = zeros(x0.n_rows,1);
	mat PC2 = zeros(as_scalar(n), as_scalar(n));
	mat Kgain;
	mat Kgain2;
	mat Kgain3;
	mat adjust2;
	Kgain = trans(solve(trans(result.yPtc), trans(Sk)));
	Kgain2 = solve(Sk,result.yPtc);
	Kgain3 = result.yPtc*inv(Sk);
	adjust2 = Kgain3*(datat - result.yt);
	xc2 = result0.yt + Kgain2*(datat - result.yt);
	PC2 = Pkp - Kgain2*Sk*trans(Kgain2);
	Kgain.print("Gain 1");
	Kgain2.print("Gain 2");
	Kgain3.print("Gain 3");
	xc2.print("xc 2:");
	PC2.print("PC 2:");
	adjust2.print("adjustment 2");

	// full UKF filter test
	FilterOut fresult;
	FilterOut fresult_enkf;
	FilterOut fresult_etkf;
	mat P0 = 0.001*diagmat(ones(2,1));
	data_new.cols(1,2).print("data matrix");
	data_new.col(0).print("time vector");
	fresult.xfilter = zeros(2, 200);
	fresult.timefilter = zeros(200, 1);
	fresult.Pfilter = zeros(2, 2, 200);
	fresult.sdfilter = zeros(2, 200);
	ukbf(obsfunc, data_new.cols(1, 2), data_new.col(0), x0, R, Q, P0, &fresult);
	
	fresult_enkf.xfilter = zeros(2, 200);
	fresult_enkf.timefilter = zeros(200, 1);
	fresult_enkf.Pfilter = zeros(2, 2, 200);
	fresult_enkf.sdfilter = zeros(2, 200);
	enkf(obsfunc, data_new.cols(1, 2), data_new.col(0), x0, R, Q, P0, &fresult_enkf);
	
	fresult_etkf.xfilter = zeros(2, 200);
	fresult_etkf.timefilter = zeros(200, 1);
	fresult_etkf.Pfilter = zeros(2, 2, 200);
	fresult_etkf.sdfilter = zeros(2, 200);
	etkf(obsfunc, data_new.cols(1, 2), data_new.col(0), x0, R, Q, P0, &fresult_etkf);
	fresult.xfilter.print("filter results:");
	//double* xfilternew = fresult.xfilter.memptr();
	//cout << "memptr lesson" << xfilternew << endl;
	//xfilternew.print("learning about memptr")


	mat residual1 = zeros(200, 1);
	residual1 = data_new.col(1) - trans(fresult.xfilter.row(0));
	residual1.print("filter vs data: UKF");

	mat residual2 = zeros(200, 1);
	residual2 = data_new.col(1) - trans(fresult_enkf.xfilter.row(0));
	residual2.print("filter vs data: EnKF");

	mat residual3 = zeros(200, 1);
	residual3 = data_new.col(1) - trans(fresult_etkf.xfilter.row(0));
	residual3.print("filter vs data: ETKF");


	mat fout = zeros(200, 2);
	fout.col(0) = data_new.col(1);
	fout.col(1) = trans(fresult.xfilter.row(0));
	fout.print("filter and data");

}

