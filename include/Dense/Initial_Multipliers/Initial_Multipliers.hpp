#ifndef FIPOPT_INITIAL_MULTIPLIERS_DENSE_HPP
#define FIPOPT_INITIAL_MULTIPLIERS_DENSE_HPP
#include <algorithm>
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective.hpp>

namespace FIPOPT::Dense
{
    template <typename Vec>
    inline Eigen::Matrix<double, Vec::RowsAtCompileTime, 1> Eval_Bounded_Initial(const MatrixBase<Vec> &x0,
                                                                                 const MatrixBase<Vec> &lb,
                                                                                 const MatrixBase<Vec> &ub,
                                                                                 const double &kappa_1,
                                                                                 const double &kappa_2)
    {
        Vec lb_peturbed = lb + Vec((lb.array().abs().max(1) * (kappa_1)).min(kappa_2 * ((ub - lb).array())));
        Vec ub_peturbed = ub - Vec((ub.array().abs().max(1) * (kappa_1)).min(kappa_2 * ((ub - lb).array())));
        Vec x0_peturbed = x0.cwiseMax(lb_peturbed).cwiseMin(ub_peturbed);
        return x0_peturbed;
    }

    template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x>
    Eigen::Matrix<double, Nh, 1> Eval_Initial_Equality_Multipliers(
        objective<Derived, Nx, Ng, Nh> &f,
        const MatrixBase<Vec_x> &x0,
        const MatrixBase<Vec_x> &zl_0 = Vec_x(Vec_x::Constant(0.)),
        const MatrixBase<Vec_x> &zu_0 = Vec_x(Vec_x::Constant(1e20)),
        const double &lbd_max = 100)
    {
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nx>;
        using Vec_A = Eigen::Matrix<double, Nx + Nh, 1>;
        using Mat_A = Eigen::Matrix<double, Nx + Nh, Nx + Nh>;
        Mat_h grad_cE = f.Eval_grad_cE(x0);

        Mat_A A(Nx + Nh, Nx + Nh);
        A.topLeftCorner(Nx, Nx).setIdentity();
        A.topRightCorner(Nh, Nx) = f.Eval_grad_cE(x0);
        A.bottomLeftCorner(Nx, Nh) = A.topRightCorner(Nh, Nx).transpose();

        Vec_A b;

        b.head(Nx) = f.Eval_grad(x0) - zl_0 + zu_0;

        Vec_A sol = A.fullPivHouseholderQr().solve(-b);

        Vec_h lbd_0 = sol.tail(Nh);

        lbd_0 = (lbd_0.array().abs() <= lbd_max).select(lbd_0, Vec_h::Constant(lbd_max));

        return lbd_0;
    }

    template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x>
    inline Eigen::Matrix<double, Ng + 2 * Nx, 1> Eval_Initial_Inequality_Multipliers(
        objective<Derived, Nx, Ng, Nh> &f,
        const MatrixBase<Vec_x> &x0,
        const double& mu,
        const double &z_max = 1e20)
    {
        using Vec_cI = Eigen::Matrix<double, Ng + 2 * Nx, 1>;
        Vec_cI z = mu * (f.Eval_cI(x0).array().inverse());
        z = (z.array() > 0).select(z, Vec_cI::Constant(1e-6));
        z = (z.array() < 1e20).select(z, Vec_cI::Constant(1e20));
        return z;
    }

}

#endif