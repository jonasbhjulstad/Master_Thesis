#ifndef FIPOPT_BARRIER_OBSERVER_HPP
#define FIPOPT_BARRIER_OBSERVER_HPP
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

        inline void Eval_f(const MatrixBase<dVec> &x, const MatrixBase<Val> &res)
        {
            static_cast<Derived *>(this)->Eval_f(x, res);
        }

        inline void Eval_f(const SparseMatrixBase<dVec> &x, const MatrixBase<Val> &res)
        {
            static_cast<Derived *>(this)->Eval_f(x, res);
        }

        inline void Eval_grad(const MatrixBase<dVec> &x, const MatrixBase<dVec> &res)
        {
            static_cast<Derived *>(this)->Eval_grad(x, res);
        }

        inline void Eval_grad(const SparseMatrixBase<dVec> &x, const MatrixBase<dVec> &res)
        {
            static_cast<Derived *>(this)->Eval_grad(x, res);
        }
 

    };
}

#endif