#ifndef FIPOPT_OBJECTIVE_JOURNALIST_HPP
#define FIPOPT_OBJECTIVE_JOURNALIST_HPP
#include <fstream>
#include <string>
#include <Common/Utils/Print.hpp>
#include <filesystem>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <array>
#include <Sparse/Functors/Objective/Objective_Observer/Objective_Observer.hpp>

namespace FIPOPT::Sparse
{
    struct objective_journalist : public objective_observer<objective_journalist>
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
        std::ofstream *Sparse_files_x[N_METHODS];
        std::ofstream *Sparse_files_data[N_METHODS];
        std::ofstream *sparse_files_x[N_METHODS];
        std::ofstream *sparse_files_data[N_METHODS];


        objective_journalist(const std::string &ID, const std::string &pwd) : ID_(ID)
        {
            namespace fs = std::filesystem;
            fs::current_path(pwd);
            fs::create_directories(ID + "/Sparse/");
            fs::create_directories(ID + "/sparse/");
            const std::string test = pwd + ID + "/Sparse/" + fnames_[0] + "_input.csv";
            for (int i; i < N_METHODS; i++)
            {
                Sparse_files_x[i] = Allocate_Open_File(pwd + ID + "/Sparse/" + fnames_[i] + "_input.csv");
                Sparse_files_data[i] = Allocate_Open_File(pwd + ID + "/Sparse/" + fnames_[i] + "_data.csv");
                sparse_files_x[i] = Allocate_Open_File(pwd + ID + "/sparse/" + fnames_[i] + "_input.csv");
                sparse_files_data[i] = Allocate_Open_File(pwd + ID + "/sparse/" + fnames_[i] + "_data.csv");
            }
        }

        inline std::ofstream *Allocate_Open_File(const std::string &fname)
        {
            return new std::ofstream(fname);
        }

        inline void Eval_f(const MatrixBase<dVec> &x, const SparseMatrixBase<spVal> &res)
        {
            Write_CSV(Sparse_files_x[0], x);
            Write_CSV(Sparse_files_data[0], res);
        }

        inline void Eval_f(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVal> &res)
        {
            // Write_CSV(sparse_files_x[0], x);
            Write_CSV(sparse_files_data[0], res);
        }

        inline void Eval_grad(const MatrixBase<dVec> &x, const SparseMatrixBase<spVec> &res)
        {
            Write_CSV(Sparse_files_x[1], x);
            Write_CSV(Sparse_files_data[1], res);
        }

        inline void Eval_grad(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &res)
        {
            // Write_CSV(sparse_files_x[1], x);
            Write_CSV(sparse_files_data[1], res);
        }

        template <typename T>
        inline void Eval_hessian_f(const MatrixBase<T> &x, const SparseMatrixBase<spMat> &res)
        {
            Write_CSV(Sparse_files_x[2], x);
            Write_CSV(Sparse_files_data[2], res);
        }

        inline void Eval_hessian_f(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spMat> &res)
        {
            // Write_CSV(sparse_files_x[2], x);
            Write_CSV(sparse_files_data[2], res);
        }

        inline void Eval_h(const MatrixBase<dVec> &x, const SparseMatrixBase<spVec> &res)
        {
            Write_CSV(Sparse_files_x[3], x);
            Write_CSV(Sparse_files_data[3], res);
        }

        inline void Eval_h(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &res)
        {
            // Write_CSV(sparse_files_x[3], x);
            Write_CSV(sparse_files_data[3], res);
        }

        inline void Eval_grad_h(const MatrixBase<dVec> &x, const SparseMatrixBase<spMat> &res)
        {
            Write_CSV(Sparse_files_x[4], x);
            Write_CSV(Sparse_files_data[4], res);
        }

        inline void Eval_grad_h(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spMat> &res)
        {

            // Write_CSV(sparse_files_x[4], x);
            Write_CSV(sparse_files_data[4], res);
        }
        inline void Eval_hessian_h(const MatrixBase<dVec> &x, const SparseMatrixBase<dVec> &lbd, const SparseMatrixBase<spMat> &res)
        {
            Write_CSV(Sparse_files_x[5], x);
            Write_CSV(Sparse_files_data[5], res);
        }

        inline void Eval_hessian_h(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &lbd, const SparseMatrixBase<spMat> &res)
        {
            // Write_CSV(sparse_files_x[5], x);
            Write_CSV(sparse_files_data[5], res);
        }

        inline void Eval_g(const MatrixBase<dVec> &x, const SparseMatrixBase<spVec> &res)
        {

            Write_CSV(Sparse_files_x[6], x);
            Write_CSV(Sparse_files_data[6], res);
        }

        inline void Eval_g(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &res)
        {
            // Write_CSV(sparse_files_x[6], x);
            Write_CSV(sparse_files_data[6], res);
        }

        inline void Eval_grad_g(const MatrixBase<dVec> &x, const SparseMatrixBase<spMat> &res)
        {

            Write_CSV(Sparse_files_x[7], x);
            Write_CSV(Sparse_files_data[7], res);
        }

        inline void Eval_grad_g(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spMat> &res)
        {
            // Write_CSV(sparse_files_x[7], x);
            Write_CSV(sparse_files_data[7], res);
        }

        template <typename T0, typename T1>
        inline void Eval_hessian_g(const MatrixBase<T0> &x, const MatrixBase<T1> &lbd_g, const MatrixBase<spMat> &res)
        {
            Write_CSV(Sparse_files_x[8], x);
            Write_CSV(Sparse_files_data[8], res);
        }

        inline void Eval_hessian_g(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &lbd_g, const SparseMatrixBase<spMat> &res)
        {
            Write_CSV(sparse_files_x[8], x);
            Write_CSV(sparse_files_data[8], res);
        }



        ~objective_journalist()
        {
            for (int i; i < N_METHODS; i++)
            {
                Sparse_files_x[i]->close();
                Sparse_files_data[i]->close();
                sparse_files_x[i]->close();
                sparse_files_data[i]->close();
            }
        }
    };
}

#endif