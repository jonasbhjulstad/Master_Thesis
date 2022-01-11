#ifndef FIPOPT_RESTORATION_LEAST_SQUARES_HPP
#define FIPOPT_RESTORATION_LEAST_SQUARES_HPP
#include <Dense/Functors/Objective/Objective.hpp>

namespace FIPOPT::Dense
{
    template <typename Derived, int Nw, int Ng_w, int Nh_w, typename Vec_x>
    inline Eigen::Matrix<double, Nw, 1> Eval_Restoration_Least_Squares(objective<Derived, Nw, Ng_w, Nh_w> &f_R,
                                                                       const double &mu,
                                                                       const double &rho,
                                                                       MatrixBase<Vec_x> &x)
    {
        constexpr static int Nx = Nw - 2 * Nh_w;
        using Vec_w = Eigen::Matrix<double, Nw, 1>;
        using Vec_cE = Eigen::Matrix<double, Nh_w, 1>;
        Vec_w w = Vec_w::Zero();
        w.head(Nx) = x;
        Vec_cE cE = f_R.Eval_cE(w);
        Vec_cE t0 = Vec_cE((mu - rho * cE.array()) / (2 * rho));
        Vec_cE t1 = Vec_cE(((mu - rho * cE.array()) / (2 * rho)).square() + mu * cE.array() / (2 * rho)).cwiseSqrt();
        w.segment(Nx, Nh_w) = t0 + t1;
        w.tail(Nh_w) = cE + w.segment(Nx, Nh_w);

        return w;
    }
}

#endif