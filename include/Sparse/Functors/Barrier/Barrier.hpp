#ifndef FIPOPT_BARRIER_DENSE_HPP
#define FIPOPT_BARRIER_DENSE_HPP

#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective.hpp>

namespace FIPOPT::Sparse
{

    template <typename Derived>
    struct barrier
    {

        inline Val operator()(const MatrixBase<dVec> &x)
        {
            return static_cast<Derived *>(this)->operator()(x);
        }

        inline dVec Eval_grad(const MatrixBase<dVec> &x)
        {
            return static_cast<Derived *>(this)->Eval_grad(x);
        }

        inline dVec Eval_cE(const MatrixBase<dVec> &x)
        {
            return static_cast<Derived *>(this)->Eval_cE(x);
        }

        inline dVec Eval_cI(const MatrixBase<dVec> &x)
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