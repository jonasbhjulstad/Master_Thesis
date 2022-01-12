#ifndef FIPOPT_BARRIER_MEMOIZED_DENSE_HPP
#define FIPOPT_BARRIER_MEMOIZED_DENSE_HPP
#include <Common/Memoizer/Memoizer.hpp>
#include <Dense/Functors/Barrier/Barrier.hpp>

namespace FIPOPT::Sparse
{
    template <int buffer_size>
    struct barrier_memoizer
    {
        using dVec = Eigen::Matrix<double, Nx, 1>;
        memoizer<dVec, Val, buffer_size> Eval;
        memoizer<dVec, dVec, buffer_size> Eval_grad;

        barrier_memoizer() : Eval("barrier_DD_Eval"),
                                         Eval_grad("barrier_DD_Eval_grad")
        {
        }
    };

    template <int buffer_size>
    struct barrier_memoizer_sparse
    {
        // Dense types
        spMat = Eigen::Matrix<double, Nx, 1>;
        memoizer<dVec, spVal, buffer_size> Eval;
        memoizer<dVec, dVec, buffer_size> Eval_grad;

        barrier_memoizer_sparse() : Eval("barrier_DS_Eval"),
                                          Eval_grad("barrier_DS_Eval_grad") {}
    };

    template <typename Derived>
    struct barrier_memoized : public barrier<barrier_memoized<Derived>>
    {
        // Dense types
        using spMat = Eigen::Matrix<double, Nx, Nx>;
        using dVec = Eigen::Matrix<double, Nx, 1>;
        using spMat = Eigen::Matrix<double, Nh, Nx>;
        using dVec = Eigen::Matrix<double, Nh, 1>;
        using spMat = Eigen::Matrix<double, Ng, Nx>;
        using dVec = Eigen::Matrix<double, Ng, 1>;

    private:
        static const int buffer_size_ = 4;
        barrier_memoizer<Nx, Ng, Nh, buffer_size_> mem_dd;
        barrier_memoizer_sparse<Nx, Ng, Nh, buffer_size_> mem_ds;

    public:
        inline Val operator()(const MatrixBase<dVec> &x)
        {
            Val res;
            if (mem_dd.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem_dd.Eval.Set_Data(x, res);
            }
            return res;
        }

        inline Val operator()(const SparseMatrixBase<dVec> &x)
        {
            Val res;
            if (mem_ds.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem_ds.Eval.Set_Data(x, res);
            }
            return res;
        }

        inline dVec Eval_grad(const MatrixBase<dVec> &x)
        {
            dVec res;
            if (mem_dd.Eval_grad.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(x);
                mem_dd.Eval_grad.Set_Data(x, res);
            }
            return res;
        }

        inline dVec Eval_grad(const SparseMatrixBase<dVec> &x)
        {
            dVec res;
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
