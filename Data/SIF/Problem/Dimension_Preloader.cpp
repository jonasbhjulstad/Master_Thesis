extern "C"
{
    #include <cutest.h>
}


#include <string>
#include <fstream>
int main(int argc, char** argv)
{
            std::string fPath = argv[1];
            fPath = fPath + "/OUTSDIF.d";
            const int funit = 42;
            int ierr;
            int status, iout, iobuffer;
            FORTRAN_open(&funit, &fPath[0], &ierr);
            int Nx = 0;
            int Nc = 0;
            int Ng = 0;
            int Nh = 0;
            CUTEST_cdimen(&status, &funit, &Nx, &Nc);
            
            double x0[Nx];
            double xlb[Nx], xub[Nx];


            if (Nc > 0)
            {
                double c[Nc], lc[Nc], uc[Nc];
                bool eqntype[Nc], eqnlinear[Nc];
                int eorder, lorder, vorder;
                eorder = 1;
                // No imposed order on linear/nonlinear constraints
                lorder = 0;
                vorder = 0;
                int nnzj, nnzh;

                CUTEST_csetup(&status, &funit, &iout, &iobuffer,
                              &Nx, &Nc, x0, xlb, xub,
                              c, lc, uc,
                              eqntype, eqnlinear, &eorder, &lorder, &vorder);

                CUTEST_cdimsj(&status, &nnzj);
                CUTEST_cdimsh(&status, &nnzh);

                for (int i = 0; i < Nc; i++)
                {
                    Nh+= eqntype[i];
                }
                Ng = Nc - Nh;
            }
            else
            {
                CUTEST_udimen(&status, &funit, &Nx);
                CUTEST_usetup(&status, &funit, &iout, &iobuffer, &Nx, x0, xlb, xub);
            }

            std::ofstream dimfile("Dimensions.hpp");
            dimfile << "#ifndef FIPOPT_DIMENSIONS_HPP\n #define FIPOPT_DIMENSIONS_HPP\n";
            dimfile << "constexpr static int GLOBAL_DIM_X = " << std::to_string(Nx) << "; \n";
            dimfile << "constexpr static int GLOBAL_DIM_G = " << std::to_string(Ng) << "; \n";
            dimfile << "constexpr static int GLOBAL_DIM_H = " << std::to_string(Nh) << "; \n";
            dimfile << "#endif";
            dimfile.close();

            std::ofstream csv_dimfile("Dimensions.csv");
            csv_dimfile << Nx << ", " << Ng << ", " << Nh;
            csv_dimfile.close();

}