#ifndef FIPOPT_BARRIER_OBSERVER_HPP
#define FIPOPT_BARRIER_OBSERVER_HPP
#include <Common/EigenDataTypes.hpp>
#include <string>

namespace FIPOPT::Dense
{
    template <typename Derived, int Nx, int Ng, int Nh>
    struct barrier_observer
    {
        // Dense types
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_g = Eigen::Matrix<double, Ng, Nx>;
        using Vec_g = Eigen::Matrix<double, Ng, 1>;

        inline void New_mu(const double& mu)
        {
            static_cast<Derived *>(this)->New_mu(mu);
        }

        inline void Eval_f(const MatrixBase<Vec_x> &x, const MatrixBase<Val> &res)
        {
            static_cast<Derived *>(this)->Eval_f(x, res);
        }

        inline void Eval_f(const SparseMatrixBase<spVec> &x, const MatrixBase<Val> &res)
        {
            static_cast<Derived *>(this)->Eval_f(x, res);
        }

        inline void Eval_grad(const MatrixBase<Vec_x> &x, const MatrixBase<Vec_x> &res)
        {
            static_cast<Derived *>(this)->Eval_grad(x, res);
        }

        inline void Eval_grad(const SparseMatrixBase<spVec> &x, const MatrixBase<Vec_x> &res)
        {
            static_cast<Derived *>(this)->Eval_grad(x, res);
        }
 

    };
}

#endif