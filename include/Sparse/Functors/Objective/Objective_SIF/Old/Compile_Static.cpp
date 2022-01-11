#include <Common/Cutest.hpp>
#include <cassert>
#include <iostream>
#include <fstream>

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        std::cout << "Usage: ./Load_parameters <filename.SIF>" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string command_1{"sifdecoder "};
    command_1+= argv[1];
    system(command_1.c_str());

    system("gfortran -c *.f");

    const std::string SIF_path{"OUTSDIF.d"};
    const std::string Parameter_File_Path{"../../include/SIF/SIF_Parameters.hpp"};

    int funit, ierr, status, Nx, Ncon;
    FORTRAN_open(&funit, SIF_path.c_str(), &ierr);
    CUTEST_cdimen(&status, &funit, &Nx, &Ncon);

    std::ofstream fParam;


    fParam.open(Parameter_File_Path, std::ofstream::out | std::ofstream::trunc);
    fParam << "#ifndef SIF_DIMENSION_PARAM_HPP\n#define SIF_DIMENSION_PARAM_HPP\n";
    fParam << "constexpr int Nx = " << Nx << ";\n";

    if (Ncon > 0)
    {

        int Ng = 0;
        int Nh = 0;
        std::cout << "The problem is constrained" << std::endl;
        std::cout << Ncon << ", " << Ng << ", " << Nh << std::endl;
        int e_order, l_order, v_order;
        // Equalities should appear before inequalities:
        e_order = 1;
        // No imposed order on linear/nonlinear constraints
        l_order = 0;
        v_order = 0;
        int iout, io_buffer, dummy;
        bool eqn_type[Ncon], eqn_linear[Ncon];
        double nullarr[Nx + Ncon];
        CUTEST_csetup(&status, &funit, &iout, &io_buffer,
                      &Nx, &Ncon, nullarr, nullarr, nullarr,
                      nullarr, nullarr, nullarr,
                      eqn_type, eqn_linear, &e_order, &l_order, &v_order);
        std::cout << Ncon << std::endl;
        for (int i = 0; i < Ncon; i++)
        {
            Nh += eqn_type[i];
        }
        std::cout << Nh << std::endl;
        Ng = Ncon - Nh;

        fParam << "constexpr int Ng = " << Ng << ";\n";
        fParam << "constexpr int Nh = " << Nh << ";\n";
    }
    else
    {

        std::cout << "The problem is unconstrained" << std::endl;

        fParam << "constexpr int Ng = " << 0 << ";\n";
        fParam << "constexpr int Nh = " << 0 << ";\n";
    }

    fParam << "#endif";
    fParam.close();
    FORTRAN_close(&funit, &ierr);
    return 0;
}