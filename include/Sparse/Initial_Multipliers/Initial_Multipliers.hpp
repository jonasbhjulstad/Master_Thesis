#ifndef FIPOPT_INITIAL_MULTIPLIERS_SPARSE_HPP
#define FIPOPT_INITIAL_MULTIPLIERS_SPARSE_HPP
#include <algorithm>
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective.hpp>

namespace FIPOPT::Sparse
{
    inline dVec Eval_Bounded_Initial(const MatrixBase<dVec> &x0,
                                     const MatrixBase<dVec> &lb,
                                     const MatrixBase<dVec> &ub,
                                     const double &kappa_1,
                                     const double &kappa_2)
    {
        dVec lb_peturbed = lb + dVec((lb.array().abs().max(1) * (kappa_1)).min(kappa_2 * ((ub - lb).array())));
        dVec ub_peturbed = ub - dVec((ub.array().abs().max(1) * (kappa_1)).min(kappa_2 * ((ub - lb).array())));
        dVec x0_peturbed = x0.cwiseMax(lb_peturbed).cwiseMin(ub_peturbed);
        return x0_peturbed;
    }

    template <typename Derived>
    dVec Eval_Initial_Equality_Multipliers(
        objective<Derived> &f,
        const MatrixBase<dVec> &x0,
        const MatrixBase<dVec> &zl_0 = dVec(dVec::Constant(0.)),
        const MatrixBase<dVec> &zu_0 = dVec(dVec::Constant(1e20)),
        const double &lbd_max = 100)
    {
        int Nx = f.Get_Nx();
        int Nh = f.Get_Nh();
        dMat grad_cE = f.Eval_grad_cE(x0);

        dMat A(Nx + Nh, Nx + Nh);
        A.topLeftCorner(Nx, Nx).setIdentity();
        A.topRightCorner(Nh, Nx) = f.Eval_grad_cE(x0);
        A.bottomLeftCorner(Nx, Nh) = A.topRightCorner(Nh, Nx).transpose();

        dVec b(Nx + Nh);

        b.head(Nx) = f.Eval_grad(x0) - zl_0 + zu_0;

        dVec sol = A.fullPivHouseholderQr().solve(-b);

        dVec lbd_0 = sol.tail(Nh);

        lbd_0 = (lbd_0.array().abs() <= lbd_max).select(lbd_0, dVec::Constant(lbd_max, Nh));

        return lbd_0;
    }

    template <typename Derived, typename dVec>
    inline Eigen::Matrix<double, Ng + 2 * Nx, 1> Eval_Initial_Inequality_Multipliers(
        objective<Derived> &f,
        const MatrixBase<dVec> &x0,
        const double &mu,
        const double &z_max = 1e20)
    {
        dVec z = mu * (f.Eval_cI(x0).array().inverse());
        z = (z.array() > 0).select(z, dVec::Constant(1e-6, z.rows()));
        z = (z.array() < 1e20).select(z, dVec::Constant(1e20, z.rows()));
        return z;
    }

}

#endif