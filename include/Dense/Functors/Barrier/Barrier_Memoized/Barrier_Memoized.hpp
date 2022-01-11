#ifndef FIPOPT_BARRIER_MEMOIZED_DENSE_HPP
#define FIPOPT_BARRIER_MEMOIZED_DENSE_HPP
#include <Common/Memoizer/Memoizer.hpp>
#include <Dense/Functors/Barrier/Barrier.hpp>

namespace FIPOPT::Dense
{
    template <int Nx, int Ng, int Nh, int buffer_size>
    struct barrier_memoizer
    {
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        memoizer<Vec_x, Val, buffer_size> Eval;
        memoizer<Vec_x, Vec_x, buffer_size> Eval_grad;

        barrier_memoizer() : Eval("barrier_DD_Eval"),
                                         Eval_grad("barrier_DD_Eval_grad")
        {
        }
    };

    template <int Nx, int Ng, int Nh, int buffer_size>
    struct barrier_memoizer_sparse
    {
        // Dense types
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        memoizer<spVec, spVal, buffer_size> Eval;
        memoizer<spVec, spVec, buffer_size> Eval_grad;

        barrier_memoizer_sparse() : Eval("barrier_DS_Eval"),
                                          Eval_grad("barrier_DS_Eval_grad") {}
    };

    template <typename Derived, int Nx, int Ng, int Nh>
    struct barrier_memoized : public barrier<barrier_memoized<Derived, Nx, Ng, Nh>, Nx, Ng, Nh>
    {
        // Dense types
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_g = Eigen::Matrix<double, Ng, Nx>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;

    private:
        static const int buffer_size_ = 4;
        barrier_memoizer<Nx, Ng, Nh, buffer_size_> mem_dd;
        barrier_memoizer_sparse<Nx, Ng, Nh, buffer_size_> mem_ds;

    public:
        inline Val operator()(const MatrixBase<Vec_x> &x)
        {
            Val res;
            if (mem_dd.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem_dd.Eval.Set_Data(x, res);
            }
            return res;
        }

        inline Val operator()(const SparseMatrixBase<Vec_x> &x)
        {
            Val res;
            if (mem_ds.Eval.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->operator()(x);
                mem_ds.Eval.Set_Data(x, res);
            }
            return res;
        }

        inline Vec_x Eval_grad(const MatrixBase<Vec_x> &x)
        {
            Vec_x res;
            if (mem_dd.Eval_grad.Get_Data(x, res) == DATA_NOT_MEMOIZED)
            {
                res = static_cast<Derived *>(this)->Eval_grad(x);
                mem_dd.Eval_grad.Set_Data(x, res);
            }
            return res;
        }

        inline Vec_x Eval_grad(const SparseMatrixBase<Vec_x> &x)
        {
            Vec_x res;
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
