#ifndef FIPOPT_OBJECTIVE_MEMOIZED_SPARSE_HPP
#define FIPOPT_OBJECTIVE_MEMOIZED_SPARSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <Sparse/Functors/Objective/Objective_Sparse.hpp>
#include <Common/Memoizer/Memoizer.hpp>

namespace FIPOPT::Sparse
{

    template <int buffer_size>
    struct NLP_memoizer_sparse_sparse
    {
        memoizer<spVec, spVal, buffer_size> Eval;
        memoizer<spVec, spVec, buffer_size> Eval_grad;
        memoizer<spVec, spMat, buffer_size> Eval_hessian;
        memoizer<spVec, spVec, buffer_size> Eval_c;
        memoizer<spVec, spMat, buffer_size> Eval_grad_c;

        NLP_memoizer_sparse_sparse() : Eval("sparse_sparse_Eval"),
                                       Eval_grad("sparse_sparse_Eval_grad"),
                                       Eval_hessian("sparse_sparse_Eval_hessian"),
                                       Eval_c("sparse_sparse_Eval_c"),
                                       Eval_grad_c("sparse_sparse_Eval_grad_c") {}
    };

    template <int Nx, int Ng, int Nh, int buffer_size>
    struct NLP_memoizer_sparse_Sparse
    {
        using spMat = Eigen::Matrix<double, Nx, Nx>;
        using spVec = Eigen::Matrix<double, Nx, 1>;
        using Mat_c = Eigen::Matrix<double, Ng + Nh, Nx>;
        using Vec_c = Eigen::Matrix<double, Ng + Nh, 1>;

        memoizer<spVec, Val, buffer_size> Eval;
        memoizer<spVec, spVec, buffer_size> Eval_grad;
        memoizer<spVec, spMat, buffer_size> Eval_hessian;
        memoizer<spVec, Vec_c, buffer_size> Eval_c;
        memoizer<spVec, Mat_c, buffer_size> Eval_grad_c;

        NLP_memoizer_sparse_Sparse() : Eval("sparse_Sparse_Eval"),
                                      Eval_grad("sparse_Sparse_Eval_grad"),
                                      Eval_hessian("sparse_Sparse_Eval_hessian"),
                                      Eval_c("sparse_Sparse_Eval_c"),
                                      Eval_grad_c("sparse_Sparse_Eval_grad_c") {}
    };

    template <typename Derived>
    struct NLP_methods_memoized_sparse : public NLP_methods_sparse<Derived>
    {
    private:
        static const int buffer_size_ = 4;
        NLP_memoizer_sparse_sparse<buffer_size_> mem_ss;
        NLP_memoizer_sparse_Sparse<buffer_size_> mem_sd;

    public:
        template <typename T>
        inline spVal operator()(const SparseMatrixBase<T> &w)
        {
            spVal res;
            if (mem_ss.Eval.Get_Data(w, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(w);
                mem_ss.Eval.Set_Data(w, res);
            }
            return res;
        }

        template <typename T>
        inline spVal operator()(const MatrixBase<T> &w)
        {
            spVal res;
            if (mem_sd.Eval.Get_Data(w, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(w);
                mem_sd.Eval.Set_Data(w, res);
            }
            return res;
        }


        template <typename T>
        inline spVec Eval_grad_sparse(const SparseMatrixBase<T> &w)
        {
            spVec res;
            if (mem_ss.Eval_grad.Get_Data(w, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(w);
                mem_ss.Eval_grad.Set_Data(w, res);
            }
            return res;
        }

        template <typename T>
        inline spVec Eval_grad_sparse(const MatrixBase<T> &w)
        {
            spVec res;
            if (mem_sd.Eval_grad.Get_Data(w, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(w);
                mem_sd.Eval_grad.Set_Data(w, res);
            }
            return res;
        }

        template <typename T>
        inline spMat Eval_hessian_sparse(const SparseMatrixBase<T> &w)
        {
            spMat res;
            if (mem_ss.Eval_hessian.Get_Data(w, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian(w);
                mem_ss.Eval_hessian.Set_Data(w, res);
            }
            return res;
        }

        template <typename T>
        inline spMat Eval_hessian_sparse(const MatrixBase<T> &w)
        {
            spMat res;
            if (mem_sd.Eval_hessian.Get_Data(w, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_hessian(w);
                mem_sd.Eval_hessian.Set_Data(w, res);
            }
            return res;
        }

        template <typename T>
        inline spVec Eval_c_sparse(const SparseMatrixBase<T> &w)
        {
            spVec res;
            if (mem_ss.Eval_c.Get_Data(w, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_c(w);
                mem_ss.Eval_c.Set_Data(w, res);
            }
            return res;
        }

        template <typename T>
        inline spVec Eval_c_sparse(const MatrixBase<T> &w)
        {
            spVec res;
            if (mem_sd.Eval_c.Get_Data(w, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_c(w);
                mem_sd.Eval_c.Set_Data(w, res);
            }
            return res;
        }

        template <typename T>
        inline spMat Eval_grad_c_sparse(const SparseMatrixBase<T> &w)
        {
            spMat res;
            if (mem_ss.Eval_grad_c.Get_Data(w, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad_c(w);
                mem_ss.Eval_grad_c.Set_Data(w, res);
            }
            return res;
        }

        template <typename T>
        inline spMat Eval_grad_c_sparse(const MatrixBase<T> &w)
        {
            spMat res;
            if (mem_sd.Eval_grad_c.Get_Data(w, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad_c(w);
                mem_sd.Eval_grad_c.Set_Data(w, res);
            }
            return res;
        }
    };

    template <typename Derived>
    struct objective_memoized_sparse : 
    public NLP_methods_memoized_sparse<Derived>, 
    public NLP_params<Derived>
    {
    };
}
#endif