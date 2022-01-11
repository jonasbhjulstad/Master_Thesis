#ifndef FIPOPT_BARRIER_OBSERVER_SPARSE_HPP
#define FIPOPT_BARRIER_OBSERVER_SPARSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <string>

namespace FIPOPT::Sparse
{
    template <typename Derived>
    struct barrier_observer
    {
        inline void New_mu(const double& mu)
        {
            static_cast<Derived *>(this)->New_mu(mu);
        }

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
 

    };
}

#endif