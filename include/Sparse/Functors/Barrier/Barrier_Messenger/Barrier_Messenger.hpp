#ifndef FIPOPT_BARRIER_MESSENGER_DENSE_HPP
#define FIPOPT_BARRIER_MESSENGER_DENSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Barrier/Barrier_Memoized/Barrier_Memoized.hpp>
#include <Dense/Functors/Barrier/Barrier_Observer/Barrier_Observer.hpp>
#include <Dense/Functors/Barrier/Barrier_Journalist/Barrier_Journalist.hpp>
#include <vector>
namespace FIPOPT::Sparse
{
    template <typename Derived, typename Derived_O>
    struct barrier_messenger : public barrier_memoized<barrier_messenger<Derived, Derived_O>>
    {
        using Observer = barrier_observer<Derived_O>;

        Observer &observer_;

        barrier_messenger(Observer &obs, const double& mu) : observer_(obs) 
        {
        }

        inline Val operator()(const MatrixBase<dVec> &x)
        {
            Val res;
            res = static_cast<Derived *>(this)->operator()(x);
            observer_.Eval_f(x, res);
            return res;
        }
        inline dVec Eval_grad(const MatrixBase<dVec> &x)
        {
            dVec res;
            res = static_cast<Derived *>(this)->Eval_grad(x);
            observer_.Eval_grad(x, res);
            return res;
        }
    };


}
#endif