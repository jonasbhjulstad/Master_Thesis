#ifndef FIPOPT_OBJECTIVE_SPARSE_HPP
#define FIPOPT_OBJECTIVE_SPARSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective_Dense.hpp>
#include <vector>
namespace FIPOPT::Dense
{
template <typename Derived>
struct NLP_methods_sparse
{
    spMat sparse_hessian;
    spMat sparse_grad_c;

    NLP_methods_sparse()
    {
        sparse_grad_c.reserve(Ng);
        for (int i = 0; i < Ng; i++)
        {
            sparse_grad_c.insert(i, Nx + i) = 1;
        }
    }


    template <typename T>
    inline spVal operator()(const MatrixBase<T> &w)
    {
        return static_cast<Derived *>(this)->Eval(w);
    }
    template <typename T>
    inline spVal operator()(const SparseMatrixBase<T> &w)
    {
        return static_cast<Derived *>(this)->Eval(w);
    }

    template <typename T>
    inline spVec Eval_grad(const MatrixBase<T> &w)
    {
        return static_cast<Derived *>(this)->Eval_grad(w);
    }

    template <typename T>
    inline spVec Eval_grad(const SparseMatrixBase<T> &w)
    {
        return static_cast<Derived *>(this)->Eval_grad(w);
    }

    template <typename T>
    inline spMat Eval_hessian(const MatrixBase<T> &w)
    {
        sparse_hessian.leftCols(Nx) = static_cast<Derived *>(this)->Eval_hessian(w);
        return sparse_hessian;
    }
    template <typename T>
    inline spMat Eval_hessian(const SparseMatrixBase<T> &w)
    {
        sparse_hessian.leftCols(Nx) = static_cast<Derived *>(this)->Eval_hessian(w);
        return sparse_hessian;
    }

    template <typename T>
    inline spVec Eval_c(const MatrixBase<T> &w)
    {
        return static_cast<Derived *>(this)->Eval_c(w);
    }
    template <typename T>
    inline spVec Eval_c(const SparseMatrixBase<T> &w)
    {
        return static_cast<Derived *>(this)->Eval_c(w);
    }

    template <typename T>
    inline spMat Eval_grad_c(const MatrixBase<T> &w)
    {
        sparse_grad_c.leftCols(Nx) = static_cast<Derived *>(this)->Eval_c(w);
        return sparse_grad_c;
    }

    template <typename T>
    inline spMat Eval_grad_c(const SparseMatrixBase<T> &w)
    {
        sparse_grad_c.leftCols(Nx) = static_cast<Derived *>(this)->Eval_c(w);
        return sparse_grad_c;
    }
};

template <typename Derived>
struct objective_sparse: 
public NLP_methods_sparse<Derived>, 
public NLP_params<Derived>{};
}
#endif