#ifndef FIPOPT_OBJECTIVE_OBSERVER_HPP
#define FIPOPT_OBJECTIVE_OBSERVER_HPP
#include <Common/EigenDataTypes.hpp>
#include <string>

namespace FIPOPT::Sparse
{
    template <typename Derived>
    struct objective_observer
    {
        using Base = Derived;

        inline void Eval_f(const MatrixBase<spVec> &x, const MatrixBase<Val> &res)
        {
            static_cast<Derived *>(this)->Eval_f(x, res);
        }

        inline void Eval_f(const SparseMatrixBase<spVec> &x, const MatrixBase<Val> &res)
        {
            static_cast<Derived *>(this)->Eval_f(x, res);
        }

        inline void Eval_grad(const MatrixBase<spVec> &x, const MatrixBase<spVec> &res)
        {
            static_cast<Derived *>(this)->Eval_grad(x, res);
        }

        inline void Eval_grad(const SparseMatrixBase<spVec> &x, const MatrixBase<spVec> &res)
        {
            static_cast<Derived *>(this)->Eval_grad(x, res);
        }

        inline void Eval_hessian_f(const MatrixBase<spVec> &x, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_f(x, res);
        }

        inline void Eval_hessian_f(const SparseMatrixBase<spVec> &x, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_f(x, res);
        }

        inline void Eval_h(const MatrixBase<spVec> &x, const MatrixBase<spVec> &res)
        {
            static_cast<Derived *>(this)->Eval_h(x, res);
        }

        inline void Eval_h(const SparseMatrixBase<spVec> &x, const MatrixBase<spVec> &res)
        {
            static_cast<Derived *>(this)->Eval_h(x, res);
        }

        inline void Eval_grad_h(const MatrixBase<spVec> &x, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_grad_h(x, res);
        }

        inline void Eval_grad_h(const SparseMatrixBase<spVec> &x, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_grad_h(x, res);
        }

        inline void Eval_hessian_h(const MatrixBase<spVec> &x, const MatrixBase<spVec> &lbd, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_h(x, lbd, res);
        }

        inline void Eval_hessian_h(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &lbd, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_h(x, lbd, res);
        }

        inline void Eval_g(const MatrixBase<spVec> &x, const MatrixBase<spVec> &res)
        {
            static_cast<Derived *>(this)->Eval_g(x, res);
        }

        inline void Eval_g(const SparseMatrixBase<spVec> &x, const MatrixBase<spVec> &res)
        {
            static_cast<Derived *>(this)->Eval_g(x, res);
        }

        inline void Eval_grad_g(const MatrixBase<spVec> &x, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_grad_g(x, res);
        }

        inline void Eval_grad_g(const SparseMatrixBase<spVec> &x, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_grad_g(x, res);
        }

        inline void Eval_hessian_g(const MatrixBase<spVec> &x, const MatrixBase<spVec> &lbd_g, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_g(x, lbd_g, res);
        }

        inline void Eval_hessian_g(const SparseMatrixBase<spVec> &x, const SparseMatrixBase<spVec> &lbd_g, const MatrixBase<spMat> &res)
        {
            static_cast<Derived *>(this)->Eval_hessian_g(x, lbd_g, res);
        }

    };
}

#endif