#include <Common/Cutest.hpp>
#include <iostream>
#include <string>
#include <memory>
#include <Eigen/Dense>
#define type *MALLOC(type, n) malloc(sizeof(type) * n)
using Vec = Eigen::VectorXd;
using Mat = Eigen::MatrixXd;
//CUTEST test functions for constrained SIF-problems
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
        std::string fname = "/home/build/FIPOPT/Data/SIF/Problem/OUTSDIF.d"; /* CUTEst data file */
        integer funit = 42;                                              /* FORTRAN unit number for OUTSDIF.d */
        integer io_buffer = 11;                                          /* FORTRAN unit for internal i/o */
        integer iout = 6;                                                /* FORTRAN unit number for error output */
        integer ierr;                                                    /* Exit flag from OPEN and CLOSE */
        integer status;                                                  /* Exit flag from CUTEst tools */
        double grad_tol = 1.e-6;                                         /* required gradient tolerance */
        integer nvar, CUTEst_ncon;
        VarTypes vtypes;
        integer ncon, ne, nzh, nnzh, nzirnh;

        integer ncon_dummy;
        doublereal *x, *bl, *bu;
        char *pname, *vnames;
        logical efirst = FALSE_, lfirst = FALSE_, nvfrst = FALSE_, grad;
        logical constrained = FALSE_;
        char *pName1 = new char[50];
        char *pName2 = new char[50];
        doublereal calls[7], cpu[4];
        integer nlin = 0, nbnds = 0, neq = 0;
        integer ExitCode;
        int i, status_cg_descent;
        doublereal time[100];
        char fgets_status;
        char line[1024];
        integer ivarty[100];
        FORTRAN_open(&funit, fname.c_str(), &ierr);
        std::cout << ierr << std::endl;
        CUTEST_cdimen(&status, &funit, &nvar, &ncon);
        integer *cType = NULL;
        MALLOC(x, nvar, doublereal);
        MALLOC(bu, nvar, doublereal);
        MALLOC(bl, nvar, doublereal);
        MALLOC(cType, ncon, integer);

        logical *equatn = NULL, *linear = NULL;
        doublereal *v = NULL, *cl = NULL, *cu = NULL;
        integer v_order = 0;
        integer l_order = 1;
        integer e_order = 0;
        double lbd[ncon];
        MALLOC(equatn, ncon + 1, logical);
        MALLOC(linear, ncon + 1, logical);
        MALLOC(v, ncon + nvar + 1, doublereal);
        MALLOC(cl, ncon + 1, doublereal);
        MALLOC(cu, ncon + 1, doublereal);
        CUTEST_csetup(&status, &funit, &iout, &io_buffer,
                      &nvar, &ncon, x, bl, bu,
                      lbd, cl, cu, equatn, linear,
                      &e_order, &l_order, &v_order);


        std::cout << "Variables: " << nvar << ", Cons: " << ncon << std::endl;
        std::cout << "x: ";
        for (int i; i < nvar; i++)
        {
                std::cout << x[i] << ", ";
        }

        std::cout << std::endl
                  << "bl: ";

        printArray(bl, ncon);

        std::cout << std::endl
                  << "bu: ";

        printArray(bu, ncon);

        std::cout
                  << "equatn: ";

        printArray(equatn, ncon);
        std::cout
                  << "linear: ";

        printArray(linear, ncon);

        std::cout << e_order << ", " << l_order << ", " << v_order << std::endl;
        int Nx_ = nvar;

        Mat hessian_g(Nx_, Nx_);
        double lbd_test[ncon];
        for (int i = 0; i < ncon; i++)
                lbd_test[i] = 1.;
        
        int ind = Nx_ + 1;
        
        double hess_g[100];
                // if (i != 0)
                //         lbd_test[i-1] = 0.;
                // lbd_test[i] = lbd[i];
        ind = i;
        CUTEST_cdhc(&status, &Nx_, &ncon, x, lbd_test, &Nx_, hess_g);

        for (int i = 0; i < Nx_; i++)
                for (int j = 0; j < Nx_; j++)
                        hessian_g(i,j) = hess_g[Nx_*i + j];



        std::cout << hessian_g << std::endl;

        lbd_test[1] = 100;

        CUTEST_cdhc(&status, &Nx_, &ncon, x, lbd_test, &Nx_, hess_g);

        for (int i = 0; i < Nx_; i++)
                for (int j = 0; j < Nx_; j++)an_g(i,j) = hess_g[Nx_*i + j];



        std::cout << hessian_g << std::endl;
}