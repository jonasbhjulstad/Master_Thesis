#ifndef FIPOPT_OBSERVER_JOURNALIST_SPARSE_HPP
#define FIPOPT_OBSERVER_JOURNALIST_SPARSE_HPP
#include <fstream>
#include <string>
#include <Common/Utils/Print.hpp>
#include <filesystem>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <array>
#include <Dense/Functors/Objective/Objective_Observer/Objective_Observer.hpp>

namespace FIPOPT::Sparse
{
    template <int Nx, int Ng, int Nh>
    struct observer_journalist : public objective_observer<observer_journalist>
    {

        const std::string ID_;
        const std::vector<std::string> fnames_ = {"Eval_f",
                                                  "Eval_grad",
                                                  "Eval_hessian_f",
                                                  "Eval_h",
                                                  "Eval_grad_h",
                                                  "Eval_hessian_h",
                                                  "Eval_g",
                                                  "Eval_grad_g",
                                                  "Eval_hessian_g"};
        static constexpr int N_METHODS = 9;
        std::ofstream file_x[N_METHODS];
        std::ofstream file_data[N_METHODS];


        observer_journalist(const std::string &ID, const std::string &pwd) : ID_(ID)
        {
            namespace fs = std::filesystem;
            fs::current_path(pwd);
            fs::create_directories(ID);
            for (int i; i < N_METHODS; i++)
            {
                file_x[i].open(pwd + ID + "/" + fnames_[i] + "_input.csv");
                file_data[i].open(pwd + ID + "/" + fnames_[i] + "_data.csv");
            }
        }

        inline std::ofstream *Allocate_Open_File(const std::string &fname)
        {
            return new std::ofstream(fname);
        }

        inline void Eval_f(const MatrixBase<dVec> &x, const MatrixBase<Val> &res)
        {
            Write_CSV(file_x[0], x);
            Write_CSV(file_data[0], res);
        }

        inline void Eval_grad(const MatrixBase<dVec> &x, const MatrixBase<dVec> &res)
        {
            Write_CSV(file_x[1], x);
            Write_CSV(file_data[1], res);
        }


        inline void Eval_hessian_f(const MatrixBase<dVec> &x, const SparseMatrixBase<spMat> &res)
        {
            Write_CSV(file_x[2], x);
            Write_CSV(file_data[2], res);
        }


        inline void Eval_h(const MatrixBase<dVec> &x, const MatrixBase<dVec> &res)
        {
            Write_CSV(file_x[3], x);
            Write_CSV(file_data[3], res);
        }


        inline void Eval_grad_h(const MatrixBase<dVec> &x, const SparseMatrixBase<spMat> &res)
        {
            Write_CSV(file_x[4], x);
            Write_CSV(file_data[4], res);
        }

        inline void Eval_hessian_h(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd, const SparseMatrixBase<spMat> &res)
        {
            Write_CSV(file_x[5], x);
            Write_CSV(file_data[5], res);
        }


        inline void Eval_g(const MatrixBase<dVec> &x, const MatrixBase<dVec> &res)
        {
            Write_CSV(file_x[6], x);
            Write_CSV(file_data[6], res);
        }


        inline void Eval_grad_g(const MatrixBase<dVec> &x, const SparseMatrixBase<spMat> &res)
        {
            Write_CSV(file_x[7], x);
            Write_CSV(file_data[7], res);
        }


        inline void Eval_hessian_g(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd_g, const SparseMatrixBase<spMat> &res)
        {
            Write_CSV(file_x[8], x);
            Write_CSV(file_data[8], res);
        }



        ~observer_journalist()
        {
            for (int i; i < N_METHODS; i++)
            {
                file_x[i].close();
                file_data[i].close();
            }
        }
    };
}

#endif