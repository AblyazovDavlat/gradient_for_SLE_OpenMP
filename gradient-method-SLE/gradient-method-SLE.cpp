// gradient-method-SLE.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <omp.h>

void initSLE(double* A, double* B, int sizeSLE)
{
	srand(time(0));
	for (int i = 0; i < sizeSLE; i++)
	{
		for (int j = i; j < sizeSLE; j++)
		{
			A[i*sizeSLE + j] = rand() % 100;
			A[j*sizeSLE + i] = A[i*sizeSLE + j];
		}
		A[i*sizeSLE + i] += 50 * sizeSLE;
		B[i] = rand() % 100;
	}
}
void initStartConditions(double* x, double* newDirectionVector,double* grad, const double* B, const int sizeSLE)
{
	for (int i = 0; i < sizeSLE; i++)
	{
		x[i] = 0;
		newDirectionVector[i] = 0;
		grad[i] = -B[i];
	}
}
void computingGradient(const double* A,const double* B,const double* x, double* grad,const int sizeSLE)
{
	for (int i = 0; i < sizeSLE; i++)
	{
		grad[i] = 0;
		for ( int j = 0; j < sizeSLE; j++)
		{
			grad[i] += A[i*sizeSLE + j] * x[j];
		}
		grad[i] = grad[i] - B[i];
	}
}
void computingDirectionVector(const double* newGrad,const double* oldGrad, double* newDirectionVector,const double* oldDirectionVector,const int sizeSLE)
{
	double newScalarProd = 0;
	double oldScalarProd = 0;
	double k;

	for (int i = 0; i < sizeSLE; i++)
	{
		newScalarProd += newGrad[i] * newGrad[i];
		oldScalarProd += oldGrad[i] * oldGrad[i];
	}
	k = newScalarProd / oldScalarProd;

	for (int i = 0; i < sizeSLE; i++)
	{
		newDirectionVector[i] = k*oldDirectionVector[i] - newGrad[i];
	}

}
double computingOffset(const double* newGrad,const double* newDirectionVector,const double* A,const int sizeSLE)
{
	double k = 0;
	double offset = 0;
	double* tmp = new double[sizeSLE];
	for (int i = 0; i < sizeSLE; i++)
	{
		tmp[i] = 0;
	}

	for (int i = 0; i < sizeSLE; i++)
	{
		k += newDirectionVector[i] * newGrad[i];
	}
	for (int i = 0; i < sizeSLE; i++)
	{
		for (int j = 0; j < sizeSLE; j++)
		{
			tmp[i] += A[i*sizeSLE + j] * newDirectionVector[j];
		}
	}
	for (int i = 0; i < sizeSLE; i++)
	{
		offset += tmp[i] * newDirectionVector[i];
	}
	offset = -k / offset;

	return offset;


}
void computingX(double* x,const double* newDirectionVector,const double offset,const int sizeSLE)
{
	for (int i = 0; i < sizeSLE; i++)
	{
		x[i] = x[i] + offset*newDirectionVector[i];
	}
}
double CheckResult(double *A, double *B, double *X, int size)
{
	double delta = 0;
	double res = 0;
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			res += A[i*size + j] * X[j];
		}
		if ((fabs(B[i] - res)) > delta){
			delta = fabs(B[i] - res);
		}
		res = 0;
	}
	return delta;
}

void computingGradientParallel(const double* A, const double* B, const double* x, double* grad, const int sizeSLE)
{
	int i, j;
#pragma omp parallel for shared(grad,A,x,B,sizeSLE) private(i,j)
	for (i = 0; i < sizeSLE; i++)
	{
		grad[i] = 0;
		for (j = 0; j < sizeSLE; j++)
		{
			grad[i] += A[i*sizeSLE + j] * x[j];
		}
		grad[i] = grad[i] - B[i];
	}
}
double computingOffsetParallel(const double* newGrad, const double* newDirectionVector, const double* A, const int sizeSLE)
{
	int i, j;
	double k = 0;
	double offset = 0;
	double* tmp = new double[sizeSLE];
	for (int i = 0; i < sizeSLE; i++)
	{
		tmp[i] = 0;
	}

	for (int i = 0; i < sizeSLE; i++)
	{
		k += newDirectionVector[i] * newGrad[i];
	}
#pragma omp parallel for shared(tmp,A,newDirectionVector,sizeSLE) private(i,j)
	for (i = 0; i < sizeSLE; i++)
	{
		for (j = 0; j < sizeSLE; j++)
		{
			tmp[i] += A[i*sizeSLE + j] * newDirectionVector[j];
		}
	}
	for (int i = 0; i < sizeSLE; i++)
	{
		offset += tmp[i] * newDirectionVector[i];
	}
	offset = -k / offset;

	return offset;
}

int _tmain(int argc, _TCHAR* argv[])
{
	int size = 1000;
	double* A = new double[size*size];
	double* B = new double[size];
	double* x = new double[size];
	double* gradient = new double[size];
	double* directionVector = new double[size];

	double* oldGradient = new double[size];
	double* oldDirectionVector = new double[size];

	double offset = 0;
	double delta = 0;
	double e = 0.00001;

	//-------------------------time------------------------------
	double t1, t2;
	//-----------------------------------------------------------

	/*A[0] = 5; A[1] = 1; A[2] = 3; A[3] = 1; A[4] = 7; A[5] = 4; A[6] = 3; A[7] = 4; A[8] = 8;
	B[0] = 4; B[1] = 7; B[2] = 12;*/
	initSLE(A, B, size);
	/*for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++)
			std::cout << A[i*size + j] << ' ';
		std::cout << std::endl;
	}
	for (int i = 0; i < size; i++)
	{
		std::cout << B[i] << std::endl;
	}*/
	
	//---------------------------------Параллельная часть-------------------------------------------
	initStartConditions(x, directionVector, gradient, B, size);
	
	t1 = omp_get_wtime();

	do{

		for (int i = 0; i < size; i++)
		{
			oldGradient[i] = gradient[i];
		}

		computingGradientParallel(A, B, x, gradient, size);
		
		/*std::cout << "Gradient :" << std::endl;
		for (int i = 0; i < size; i++)
		{
			std::cout << gradient[i] << ' ';
		}
		std::cout << std::endl;*/
			

		for (int i = 0; i < size; i++)
		{
			oldDirectionVector[i] = directionVector[i];
		}

		computingDirectionVector(gradient, oldGradient, directionVector, oldDirectionVector, size);

		/*std::cout << "directionVector :" << std::endl;
		for (int i = 0; i < size; i++)
		{
			std::cout << directionVector[i] << ' ';
		}
		std::cout << std::endl;*/

		offset = computingOffsetParallel(gradient, directionVector, A, size);

	//	std::cout << "Offset :" << offset << std::endl;
		

		computingX(x, directionVector, offset, size);
		delta = CheckResult(A, B, x, size);

		/*for (int i = 0; i < size; i++)
		{
			std::cout << x[i] << ' ';
		}*/
		std::cout <<"Delta:"<<delta<< std::endl;
		std::cout << std::endl;
		

	} while (delta > e);

	t2 = omp_get_wtime();

	std::cout <<"Time by parallel method :"<< t2 - t1 << std::endl;
//------------------------------------------------------------------------------------------------

//---------------------------------Последовательная часть-----------------------------------------
	
	initStartConditions(x, directionVector, gradient, B, size);

	t1 = omp_get_wtime();

	do{

		for (int i = 0; i < size; i++)
		{
			oldGradient[i] = gradient[i];
		}

		computingGradient(A, B, x, gradient, size);

		/*std::cout << "Gradient :" << std::endl;
		for (int i = 0; i < size; i++)
		{
		std::cout << gradient[i] << ' ';
		}
		std::cout << std::endl;*/


		for (int i = 0; i < size; i++)
		{
			oldDirectionVector[i] = directionVector[i];
		}

		computingDirectionVector(gradient, oldGradient, directionVector, oldDirectionVector, size);

		/*std::cout << "directionVector :" << std::endl;
		for (int i = 0; i < size; i++)
		{
		std::cout << directionVector[i] << ' ';
		}
		std::cout << std::endl;*/

		offset = computingOffset(gradient, directionVector, A, size);

		//std::cout << "Offset :" << offset << std::endl;


		computingX(x, directionVector, offset, size);
		delta = CheckResult(A, B, x, size);

		/*for (int i = 0; i < size; i++)
		{
		std::cout << x[i] << ' ';
		}*/
		std::cout << "Delta:" << delta << std::endl;
		std::cout << std::endl;


	} while (delta > e);

	t2 = omp_get_wtime();

	std::cout << "Time by consistent method :"<< t2 - t1<< std::endl;
//-----------------------------------------------------------------------------------

	delete[] A;
	delete[] B;
	delete[] x;
	delete[] gradient;
	delete[] directionVector;
	delete[] oldGradient;
	delete[] oldDirectionVector;


	return 0;
}

