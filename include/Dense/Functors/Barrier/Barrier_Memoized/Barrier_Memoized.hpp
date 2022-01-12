#ifndef FIPOPT_BARRIER_MEMOIZED_DENSE_HPP
#define FIPOPT_BARRIER_MEMOIZED_DENSE_HPP
#include <Common/Memoizer/Memoizer.hpp>
#include <Dense/Functors/Barrier/Barrier.hpp>

namespace FIPOPT::Sparse
{
    template <int buffer_size>
    struct barrier_memoizer
    {
        memoizer<dVec, Val, buffer_size> Eval;
        memoizer<dVec, dVec, buffer_size> Eval_grad;

        barrier_memoizer() : Eval("barrier_Eval"),
                                         Eval_grad("barrier_Eval_grad")
        {
        }
    };


    template <typename Derived>
    struct barrier_memoized : public barrier<barrier_memoized<Derived>>
    {

    private:
        static const int buffer_size_ = 4;
        barrier_memoizer<buffer_size_> mem;

    public:
        inline Val operator()(const MatrixBase<dVec> &x)
        {
            Val res;
            if (mem.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem.Eval.Set_Data(x, res);
            }
            return res;
        }


        inline dVec Eval_grad(const MatrixBase<dVec> &x)
        {
            dVec res;
            if (mem.Eval_grad.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(x);
                mem.Eval_grad.Set_Data(x, res);
            }
            return res;
        }


        inline double Get_mu()
        {
            return static_cast<Derived*>(this)->Get_mu();
        }
    };
}

#endif
