#ifndef FIPOPT_PRINT_HPP
#define FIPOPT_PRINT_HPP
#include <Common/EigenDataTypes.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
namespace FIPOPT
{

    void inline Print_Iteration_Steps(const int &i, const double &alpha, const double &alpha_z_g, const double &alpha_z_lb, const double &alpha_z_ub)
    {
        std::cout << "Iteration " << i << std::endl;
        std::cout << "Alpha: \t " << alpha << ", alpha_z_g: \t" << alpha_z_g << ", alpha_z_ub: \t" << alpha_z_ub << ", alpha_z_lb: \t" << alpha_z_lb << std::endl;
    }

    template <typename Vec_x, typename Vec_cE, typename Vec_cI>
    void inline Print_Iteration_States(const Vec_x &x, Vec_cE &lbd, const Vec_cI &z)
    {
        std::cout << "x:\t" << x.transpose() << std::endl;
        std::cout << "lbd:\t" << lbd.transpose() << std::endl;
        std::cout << "z:\t" << z.transpose() << std::endl;
    }
    template <typename T>
    inline void Write_CSV(std::ofstream &file, const MatrixBase<T> &data)
    {

        static const Eigen::IOFormat CSVFormat = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
        file << data.transpose().format(CSVFormat) << '\n';
        file.flush();
    }

    template <typename T>
    inline void Write_CSV(std::ofstream *file, const SparseMatrixBase<T> &data)
    {
        *file << data.transpose() << '\n';
        file->flush();
    }


    struct CSV_iteration_journalist
    {
        // using namespace Eigen;
        CSV_iteration_journalist(const std::string &pwd, const std::string dest)
        {
            namespace fs = std::filesystem;
            fs::current_path(pwd);
            fs::create_directories(dest);

            file_x.open(pwd + dest + "x.csv");
            file_lbd.open(pwd + dest + "lbd.csv");
            file_z.open(pwd + dest + "z.csv");
        }

        template <typename Vec_x, typename Vec_cE, typename Vec_cI>
        inline void Write(const Vec_x &x, const Vec_cE &lbd, const Vec_cI &z)
        {
            const Eigen::IOFormat CSVFormat = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");
            file_x << x.transpose().format(CSVFormat) << '\n';
            file_lbd << lbd.transpose().format(CSVFormat) << '\n';
            file_z << z.transpose().format(CSVFormat) << '\n';
        }

        ~CSV_iteration_journalist()
        {
            file_x.close();
            file_lbd.close();
            file_z.close();
        }

    private:
        std::ofstream file_x;
        std::ofstream file_z;
        std::ofstream file_lbd;
    };
}
#endif