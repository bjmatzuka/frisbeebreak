#include "stdafx.h"
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;

void testnew1() {
	// basic test 
	double T = 3.14;
	cout << T;

	// armadillo test
	mat A(2, 3);
	A(0, 1) = 456.0;  // directly access an element (indexing starts at 0)
	A.print("A:");

	// observation error covariance
	mat R = eye<mat>(3, 3);
	R.print("R:");

	// system error covariance
	mat Q = eye<mat>(3, 3);
	Q.print("Q:");

	// initial conditions for state vector
	vec ic;
	ic = ones(Q.n_rows,1);
	ic.print("ic:");

	// data file
	mat data;
	data.zeros(2, 20);
	data.print("data:");


}