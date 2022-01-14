#ifndef FIPOPT_BARRIER_JOURNALIST_SPARSE_HPP
#define FIPOPT_BARRIER_JOURNALIST_SPARSE_HPP
#include <fstream>
#include <string>
#include <Common/Utils/Print.hpp>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <array>
// #include <Common/EigenDataTypes.hpp>
#include <Sparse/Functors/Barrier/Barrier_Observer/Barrier_Observer.hpp>

namespace FIPOPT::Sparse
{
    struct barrier_journalist : public barrier_observer<barrier_journalist>
    {


        const std::string ID_;
        const std::string pwd_;
        const std::vector<std::string> fnames_ = {"Eval_f",
                                                  "Eval_grad"};
        static constexpr int N_METHODS = 2;
        std::ofstream *Sparse_files_x[N_METHODS];
        std::ofstream *Sparse_files_data[N_METHODS];
        std::ofstream *sparse_files_x[N_METHODS];
        std::ofstream *sparse_files_data[N_METHODS];
        std::ofstream f_mu_;

        const Eigen::IOFormat CSVFormat = Eigen::IOFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

        barrier_journalist(const std::string &ID, const std::string &pwd) : ID_(ID), pwd_(pwd)
        {
            namespace fs = std::filesystem;
            fs::current_path(pwd);
            fs::create_directories(ID + "/Sparse/");
            fs::create_directories(ID + "/sparse/");
            f_mu_.open(ID + "/mu.csv");
        }

        void New_mu(const double &mu)
        {

            std::string mu_str = std::to_string(mu);
            namespace fs = std::filesystem;
            fs::current_path(pwd_);
            fs::create_directories(ID_ + "/Sparse/" + mu_str + "/");
            fs::create_directories(ID_ + "/sparse/" + mu_str + "/");
            f_mu_ << mu << '\n';
            for (int i; i < N_METHODS; i++)
            {
                Sparse_files_x[i] = Allocate_Open_File(pwd_ + ID_ + "/Sparse/" + mu_str + "/" + fnames_[i] + "_input.csv");
                Sparse_files_data[i] = Allocate_Open_File(pwd_ + ID_ + "/Sparse/" + mu_str + "/"  + fnames_[i] + "_data.csv");
                sparse_files_x[i] = Allocate_Open_File(pwd_ + ID_ + "/sparse/" + mu_str + "/" + fnames_[i] + "_input.csv");
                sparse_files_data[i] = Allocate_Open_File(pwd_ + ID_ + "/sparse/" + mu_str + "/" + fnames_[i] + "_data.csv");
            }
        }

        inline std::ofstream *Allocate_Open_File(const std::string &fname)
        {
            return new std::ofstream(fname);
        }

        template <typename T>
        inline void Eval_f(const MatrixBase<T> &x, const SparseMatrixBase<spVal> &res)
        {
            Write_CSV(Sparse_files_x[0], x);
            Write_CSV(Sparse_files_data[0], res);
        }

        inline void Eval_f(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVal> &res)
        {
            Write_CSV(sparse_files_x[0], x);
            Write_CSV(sparse_files_data[0], res);
        }

        template <typename T>
        inline void Eval_grad(const MatrixBase<T> &x, const SparseMatrixBase<spVec> &res)
        {
            Write_CSV(Sparse_files_x[1], x);
            Write_CSV(Sparse_files_data[1], res);
        }

        inline void Eval_grad(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &res)
        {
            // Write_CSV(sparse_files_x[1], x);
            Write_CSV(sparse_files_data[1], res);
        }

        ~barrier_journalist()
        {
            for (int i; i < N_METHODS; i++)
            {
                Sparse_files_x[i]->close();
                Sparse_files_data[i]->close();
                sparse_files_x[i]->close();
                sparse_files_data[i]->close();
            }
            f_mu_.close();
        }
    };
}

#endif