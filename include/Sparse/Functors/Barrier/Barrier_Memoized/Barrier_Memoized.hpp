#ifndef FIPOPT_BARRIER_MEMOIZED_Sparse_HPP
#define FIPOPT_BARRIER_MEMOIZED_Sparse_HPP
#include <Common/Memoizer/Memoizer.hpp>
#include <Sparse/Functors/Barrier/Barrier.hpp>

namespace FIPOPT::Sparse
{
    template <int buffer_size>
    struct barrier_memoizer_dense
    {
        memoizer<dVec, spVal, buffer_size> Eval;
        memoizer<dVec, spVec, buffer_size> Eval_grad;

        barrier_memoizer_dense() : Eval("barrier_DS_Eval"),
                                         Eval_grad("barrier_DS_Eval_grad")
        {
        }
    };

    template <int buffer_size>
    struct barrier_memoizer_sparse
    {
        memoizer<spVec, spVal, buffer_size> Eval;
        memoizer<spVec, spVec, buffer_size> Eval_grad;

        barrier_memoizer_sparse() : Eval("barrier_SS_Eval"),
                                          Eval_grad("barrier_SS_Eval_grad") {}
    };

    template <typename Derived>
    struct barrier_memoized : public barrier<barrier_memoized<Derived>>
    {


    private:
        static const int buffer_size_ = 4;
        barrier_memoizer_dense<buffer_size_> mem_dd;
        barrier_memoizer_sparse<buffer_size_> mem_ds;

    public:
        inline Val operator()(const MatrixBase<spVec> &x)
        {
            Val res;
            if (mem_dd.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem_dd.Eval.Set_Data(x, res);
            }
            return res;
        }

        inline Val operator()(const SparseMatrixBase<spVec> &x)
        {
            Val res;
            if (mem_ds.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem_ds.Eval.Set_Data(x, res);
            }
            return res;
        }

        inline spVec Eval_grad(const MatrixBase<spVec> &x)
        {
            spVec res;
            if (mem_dd.Eval_grad.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(x);
                mem_dd.Eval_grad.Set_Data(x, res);
            }
            return res;
        }

        inline spVec Eval_grad(const SparseMatrixBase<spVec> &x)
        {
            spVec res;
            if (mem_ds.Eval_grad.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(x);
                mem_ds.Eval_grad.Set_Data(x, res);
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
