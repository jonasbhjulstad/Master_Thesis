#ifndef FIPOPT_JOURNALIST_DENSE_HPP
#define FIPOPT_JOURNALIST_DENSE_HPP
#include <fstream>
#include <string>
#include <Common/Utils/Print.hpp>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <array>
// #include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Barrier/Barrier_Observer/Barrier_Observer.hpp>

namespace FIPOPT::Sparse
{
    template <int Nx, int Ng, int Nh>
    struct barrier_journalist : public barrier_observer<barrier_journalist<Nx, Ng, Nh>>
    {

        const std::string ID_;
        const std::string pwd_;
        const std::vector<std::string> fnames_ = {"Eval_f",
                                                  "Eval_grad"};
        static constexpr int N_METHODS = 2;
        std::ofstream files_x[N_METHODS];
        std::ofstream files_data[N_METHODS];
        std::ofstream f_mu_;

        const Eigen::IOFormat CSVFormat = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

        barrier_journalist(const std::string &ID, const std::string &pwd) : ID_(ID), pwd_(pwd)
        {
            namespace fs = std::filesystem;
            fs::current_path(pwd);
            fs::create_directories(ID + "/dense/");
            fs::create_directories(ID + "/sparse/");
            f_mu_.open(ID + "/mu.csv");
        }

        void New_mu(const double &mu)
        {

            std::string mu_str = std::to_string(mu);
            namespace fs = std::filesystem;
            fs::current_path(pwd_);
            fs::create_directories(ID_ + "/dense/" + mu_str + "/");
            f_mu_ << mu << '\n';
            for (int i; i < N_METHODS; i++)
            {
                files_x[i] = Allocate_Open_File(pwd_ + ID_ + "/dense/" + mu_str + "/" + fnames_[i] + "_input.csv");
                files_data[i] = Allocate_Open_File(pwd_ + ID_ + "/dense/" + mu_str + "/"  + fnames_[i] + "_data.csv");
            }
        }

        inline std::ofstream *Allocate_Open_File(const std::string &fname)
        {
            return new std::ofstream(fname);
        }

        inline void Eval_f(const MatrixBase<dVec> &x, const MatrixBase<Val> &res)
        {
            Write_CSV(files_x[0], x);
            Write_CSV(files_data[0], res);
        }


        inline void Eval_grad(const MatrixBase<dVec> &x, const MatrixBase<dVec> &res)
        {
            Write_CSV(files_x[1], res);
            Write_CSV(files_data[1], res);
        }

        ~barrier_journalist()
        {
            for (int i; i < N_METHODS; i++)
            {
                files_x[i].close();
                files_data[i].close();
            }
            f_mu_.close();
        }
    };
}

#endif