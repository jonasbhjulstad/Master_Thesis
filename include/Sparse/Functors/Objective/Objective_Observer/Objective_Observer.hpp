#ifndef FIPOPT_OBJECTIVE_OBSERVER_HPP
#define FIPOPT_OBJECTIVE_OBSERVER_HPP
#include <Common/EigenDataTypes.hpp>
#include <string>

namespace FIPOPT::Sparse
{
    template <typename Derived>
    struct objective_observer
    {
        // Dense types

        using Base = Derived;

        inline void Eval_f(const MatrixBase<dVec> &x, const MatrixBase<Val> &res)
        {
            static_cast<Derived *>(this)->Eval_f(x, res);
        }

        inline void Eval_grad(const MatrixBase<dVec> &x, const MatrixBase<dVec> &res)
        {
            static_cast<Derived *>(this)->Eval_grad(x, res);
        }

        inline void Eval_hessian_f(const MatrixBase<dVec> &x, const SparseMatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_f(x, res);
        }

        inline void Eval_h(const MatrixBase<dVec> &x, const MatrixBase<dVec> &res)
        {
            static_cast<Derived *>(this)->Eval_h(x, res);
        }

        inline void Eval_grad_h(const MatrixBase<dVec> &x, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_grad_h(x, res);
        }

        inline void Eval_hessian_h(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd, const SparseMatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_h(x, lbd, res);
        }

        inline void Eval_g(const MatrixBase<dVec> &x, const MatrixBase<dVec> &res)
        {
            static_cast<Derived *>(this)->Eval_g(x, res);
        }

        inline void Eval_grad_g(const MatrixBase<dVec> &x, const SparseMatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_grad_g(x, res);
        }

        inline void Eval_hessian_g(const MatrixBase<dVec> &x, const MatrixBase<dVec> &lbd_g, const SparseMatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_g(x, lbd_g, res);
        }
    };
}

#endif