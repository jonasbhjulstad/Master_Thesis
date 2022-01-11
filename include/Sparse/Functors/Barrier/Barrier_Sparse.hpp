#ifndef FIPOPT_OBJECTIVE_BARRIER_SPARSE_HPP
#define FIPOPT_OBJECTIVE_BARRIER_SPARSE_HPP

#include <Common/EigenDataTypes.hpp>
#include <Sparse/Functors/Objective/Objective_Sparse.hpp>
#include <Sparse/Functors/Objective/Objective_Sparse.hpp>
#include <Sparse/Functors/Barrier/Objective_Barrier.hpp>
|
namespace FIPOPT::Sparse
{
    template <typename Derived>
    struct objective_barrier_sparse: public objective_barrier<Derived_B>
    {
    private:
        using spVec = Eigen::Matrix<double, Nx, 1>;
        using Vec_s = Eigen::Matrix<double, Ng, 1>;
        using spVec = Eigen::Matrix<double, Nx + Ng, 1>;

    protected:
        objective_sparse<Derived> *f_;
        const double mu_;

    public:
        objective_barrier_sparse(objective_sparse<Derived> &f, const double &mu) : f_(&f), mu_(mu) {}

        template <typename BaseType>
        inline spVal operator()(const BaseType &w)
        {
            // spVec x = spVec(w).head(Nx);
            // Vec_s s = spVec(w).tail(Ng);

            // Val con_val = Val(-mu_ * Vec_s(s).array().log().sum());
            // Val bounds_val = Val(-mu_ * ((x - f_->Get_x_lb()).array().log().sum() + (f_->Get_x_ub() - x).array().log().sum()));

            return;
        }

        // template <typename T>
        // inline double operator()(const SparseMatrixBase<T> &x)
        // {
        //     return operator()(x.eval());
        // }

        template <typename BaseType>
        inline spVec Eval_grad(const BaseType &w)
        {
            spVec x = spVec(w).head(Nx);
            Vec_s s = spVec(w).tail(Ng);

            Vec_s res(Nx + Ng);

            res.head(Nx) = f_->Eval_grad(x).array() - mu_ * spVec(x).array().inverse();
            res.tail(Ng) = -mu_ * s.array().inverse();

            return res;
        }

        template <typename BaseType>
        inline spVec Eval_c(const BaseType &w)
        {
            return f_->Eval_c(w);
        }

        // template <typename T>
        // inline spVec Eval_grad(const SparseMatrixBase<T> &x)
        // {
        //     spVec logBarrier = x;
        //     for (spVec::InnerIterator it(logBarrier); it; ++it)
        //     {
        //             logBarrier.insert(it.col()) = mu_/x(it.col());
        //     }

        //     return f_->Eval_grad(x) - logBarrier;
        // }

        inline double Get_mu()
        {
            return mu_;
        }
    };
}

#endif