#ifndef FIPOPT_BARRIER_DENSE_HPP
#define FIPOPT_BARRIER_DENSE_HPP

#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective.hpp>

namespace FIPOPT::Dense
{

    template <typename Derived, int Nx, int Ng, int Nh>
    struct barrier
    {
        // Dense types
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Vec_cE = Eigen::Matrix<double, Nh, 1>;
        using Vec_cI = Eigen::Matrix<double, Ng + 2 * Nx, 1>;

        inline Val operator()(const MatrixBase<Vec_x> &x)
        {
            return static_cast<Derived *>(this)->operator()(x);
        }

        inline Vec_x Eval_grad(const MatrixBase<Vec_x> &x)
        {
            return static_cast<Derived *>(this)->Eval_grad(x);
        }

        inline Vec_cE Eval_cE(const MatrixBase<Vec_x> &x)
        {
            return static_cast<Derived *>(this)->Eval_cE(x);
        }

        inline Vec_cI Eval_cI(const MatrixBase<Vec_x> &x)
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