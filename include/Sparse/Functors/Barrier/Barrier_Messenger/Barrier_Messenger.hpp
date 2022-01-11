#ifndef FIPOPT_BARRIER_MESSENGER_Sparse_HPP
#define FIPOPT_BARRIER_MESSENGER_Sparse_HPP
#include <Common/EigenDataTypes.hpp>
#include <Sparse/Functors/Barrier/Barrier_Memoized/Barrier_Memoized.hpp>
#include <Sparse/Functors/Barrier/Barrier_Observer/Barrier_Observer.hpp>
#include <Sparse/Functors/Barrier/Barrier_Messenger/Barrier_Messenger.hpp>
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

        inline Val operator()(const MatrixBase<spVec> &x)
        {
            Val res;
            res = static_cast<Derived *>(this)->operator()(x);
            observer_.Eval_f(x, res);
            return res;
        }

        inline Val operator()(const SparseMatrixBase<spVec> &x)
        {
            Val res;
            res = static_cast<Derived *>(this)->operator()(x);
            observer_.Eval_f(x, res);
            return res;
        }

        inline spVec Eval_grad(const MatrixBase<spVec> &x)
        {
            spVec res;
            res = static_cast<Derived *>(this)->Eval_grad(x);
            observer_.Eval_grad(x, res);
            return res;
        }

        inline spVec Eval_grad(const SparseMatrixBase<spVec> &x)
        {
            spVec res;
            res = static_cast<Derived *>(this)->Eval_grad(x);
            observer_.Eval_grad(x, res);
            return res;
        }
    };


}
#endif