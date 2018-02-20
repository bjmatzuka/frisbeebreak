// cvode_test.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <cstdio>

// forward declare armadillo function instead of making a header file for it
void runArmadillo();
void testnew1();


extern "C" {
	int runTest();
}
int ODEtest1();


int _tmain(int argc, _TCHAR* argv[])
{
	char ret;
	std::printf("Beginning CVODE Test");
	ODEtest1();
	std::printf("End CVODE Test\n Press any key to continue...\n");
	std::scanf("%c", &ret);
	std::printf("Beginning Test1\n");
	testnew1();
	std::printf("\n End Test1 \n Press any key to continue...\n");
	std::scanf("%c", &ret);
	std::printf("Beginning Armadillo Test"); 
	runArmadillo();
	std::printf("End Armadillo Test\n Press any key to continue...\n");
	std::scanf("%c", &ret);
	
	return 0;
}
