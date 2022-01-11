#include "Eigen/Dense"
#include <Common/EigenDataTypes.hpp>
#include "cutest.h"
#include <iostream>
#include <string.h>
#include <memory>

#define type *MALLOC(type, n) malloc(sizeof(type) * n)
//CUTEST test functions for unconstrained SIF-problems
//OUTSDIF-path needs to be configured

template <typename T>
void printArray(T x, integer n)
{
        for (int i = 0; i < n; i++)
        {
                std::cout << x[i] << ", ";
        }
        std::cout << std::endl;
}

int main()

{
        char *fname = "/home/deb/Downloads/cutest/cutest/sif/uncsif/OUTSDIF.d"; /* CUTEst data file */
        integer funit = 42;                                                     /* FORTRAN unit number for OUTSDIF.d */
        integer io_buffer = 11;                                                 /* FORTRAN unit for internal i/o */
        integer iout = 6;                                                       /* FORTRAN unit number for error output */
        integer ierr;                                                           /* Exit flag from OPEN and CLOSE */
        integer status;                                                         /* Exit flag from CUTEst tools */
        integer nvar;

        doublereal *x, *bl, *bu;
        FORTRAN_open(&funit, fname, &ierr);
        CUTEST_udimen(&status, &funit, &nvar);
        MALLOC(x, nvar, doublereal);
        MALLOC(bu, nvar, doublereal);
        MALLOC(bl, nvar, doublereal);

        CUTEST_usetup(&status, &funit, &iout, &io_buffer,
                      &nvar, x, bl, bu);
        
        integer nnzh;
        // /* Unconstrained dimensioning and report routines */
        CUTEST_udimsh(&status, &nnzh);
        // CUTEST_udimse(&status, &ne, &nzh,
        //               &nzirnh);
        // CUTEST_uvartype(&status, &n, &ivarty[0]);
        // CUTEST_unames(&status, &n, pName1,
        //               pName2);
        // CUTEST_ureport(&status,calls,&time[0]);

        // CUTEST_cdimen( &status, &funit, &nvar, &ncon) ;
        doublereal* h;
        integer *irnh, *icnh, lh;
        MALLOC(h, nnzh, doublereal);
        MALLOC(irnh, nnzh, integer);
        MALLOC(icnh, nnzh, integer);
        // MALLOC(lh, nnzg, integer);
        CUTEST_ush(&status, &nvar, x, &nnzh, &lh, h, irnh, icnh);

        std::cout << "Variables: " << nvar << std::endl;
        std::cout << "x: ";
        for (int i = 0; i < nvar; i++)
        {
                std::cout << x[i] << ", ";
        }

        std::cout << std::endl << "bl: ";

        printArray(bl, nvar);

        std::cout
                  << "bu: ";
        printArray(bu, nvar);

        std::cout << "nnzh: " << nnzh << std::endl;

        std::cout << "lh: " << lh << std::endl;
        std::cout << "h: ";
        printArray(h, nnzh);
        std::cout << "irnh: ";
        printArray(irnh, nnzh);

        std::cout << "icnh: ";
        printArray(icnh, nnzh);


        std::vector<Triplet> hessianTriplets;
        for (int i = 0; i < nnzh; i++)
        {
                std::cout << irnh[i] << ", " << icnh[i] << ", " << h[i] << std::endl;
                hessianTriplets.push_back(Triplet(irnh[i]-1, icnh[i]-1, h[i]));
        }

        spMat hessian(nvar, nvar);
        hessian.setFromTriplets(hessianTriplets.begin(), hessianTriplets.end());

        std::cout << hessian << std::endl;
        

        



}