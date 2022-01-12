#ifndef FIPOPT_RESTORATION_LEAST_SQUARES_HPP
#define FIPOPT_RESTORATION_LEAST_SQUARES_HPP
#include <Sparse/Functors/Objective/Objective.hpp>

namespace FIPOPT::Sparse
{
    template <typename Derived>
    inline dVec Eval_Restoration_Least_Squares(objective<Derived> &f_R,
                                                                       const double &mu,
                                                                       const double &rho,
                                                                       MatrixBase<dVec> &x)
    {
        int Nw = f_R.Get_Nx();
        int Nh_w = f_R.Get_Nh();
        int Nx = Nw - 2*Nh_w;
        dVec w = dVec::Constant(0., Nw);
        w.head(Nx) = x;
        dVec cE = f_R.Eval_cE(w);
        dVec t0 = dVec((mu - rho * cE.array()) / (2 * rho));
        dVec t1 = dVec(((mu - rho * cE.array()) / (2 * rho)).square() + mu * cE.array() / (2 * rho)).cwiseSqrt();
        w.segment(Nx, Nh_w) = t0 + t1;
        w.tail(Nh_w) = cE + w.segment(Nx, Nh_w);

        return w;
    }
}

#endif