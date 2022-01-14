#ifndef FIPOPT_BARRIER_Sparse_HPP
#define FIPOPT_BARRIER_Sparse_HPP

#include <Common/EigenDataTypes.hpp>
#include <Sparse/Functors/Objective/Objective.hpp>

namespace FIPOPT::Sparse
{

    template <typename Derived>
    struct barrier
    {

        template <typename T>
        inline Val operator()(const MatrixBase<T> &x)
        {
            return static_cast<Derived *>(this)->operator()(x);
        }

        template <typename T>
        inline Val operator()(const SparseMatrixBase<T> &x)
        {
            return static_cast<Derived *>(this)->operator()(x);
        }

        template <typename T>
        inline spVec Eval_grad(const MatrixBase<T> &x)
        {
            return static_cast<Derived *>(this)->Eval_grad(x);
        }

        template <typename T>
        inline spVec Eval_grad(const SparseMatrixBase<T> &x)
        {
            return static_cast<Derived *>(this)->Eval_grad(x);
        }

        template <typename T>
        inline spVec Eval_cE(const MatrixBase<T> &x)
        {
            return static_cast<Derived *>(this)->Eval_cE(x);
        }

        template <typename T>
        inline spVec Eval_cE(const SparseMatrixBase<T> &x)
        {
            return static_cast<Derived *>(this)->Eval_cE(x);
        }

        template <typename T>
        inline spVec Eval_cI(const MatrixBase<T> &x)
        {
            return static_cast<Derived *>(this)->Eval_cI(x);
        }

        template <typename T>
        inline spVec Eval_cI(const SparseMatrixBase<T> &x)
        {
            return static_cast<Derived *>(this)->Eval_cI(x);
        }

        inline double Get_mu()
        {
            return static_cast<Derived*>(this)->Get_mu();
        }
    };

}

#endif