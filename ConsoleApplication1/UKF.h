#pragma once

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

#ifdef UKF_EXPORTS  
#define UKF_API extern "C" __declspec(dllexport)   
#define UKF_API_CPP __declspec(dllexport)   
#else  
#define UKF_API extern "C" __declspec(dllimport)   
#define UKF_API_CPP __declspec(dllimport)   
#endif  

using namespace std;
using namespace arma;

struct UTdataOut {
	mat yt;
	mat yPt;
	mat yPtc;
};

struct FilterOut {
	mat xfilter;
	vec timefilter;
	cube Pfilter;
	mat sdfilter;
};

UKF_API_CPP void UTpred(mat x, mat PP, vec time, UTdataOut *result);
UKF_API_CPP void UTdata(mat obsf, mat y, mat P, mat obsvec, UTdataOut *result);
UKF_API_CPP void ukbf(mat obsf, mat data, vec time, vec x0, mat R, mat Q, mat P0, FilterOut *result);

UKF_API void MatMultiply(double* a, double* b, double* c, double* N);

//UKF_API void UTpred_R(double* x, double* PP, double* N, double* time, double* yt, double* yPt, double* yPtc);
UKF_API void ukbf_R(double* obsf, double* data, long* M, double* time, long* N, double* x0, long* L,
	double* R, double* Q, double* P0,
	double* xfilter, double* timefilter, double* Pfilter, double* sdfilter);
//UKF_API void UTdata_R(double* obsf, double* y, double* P, double* obsvec, double* yt, double* yPt, double* yPtc);

