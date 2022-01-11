#ifndef FIPOPT_RESTORATION_LEAST_SQUARES_HPP
#define FIPOPT_RESTORATION_LEAST_SQUARES_HPP
#include <Sparse/Functors/Objective/Objective.hpp>

namespace FIPOPT::Sparse
{
    template <typename Derived>
    inline void Eval_Restoration_Least_Squares(objective<Derived> &f_R,
                                                const double &mu,
                                                const double &rho,
                                                spVec& w)
    {
        const int Nh_w = f_R.Get_Nh();
        const int Nw = f_R.Get_Nx();
        const int Nx = Nw - 2*Nh_w;
        
        spVec cE = f_R.Eval_cE(w);
        spVec t0(Nw);
        spVec t1(Nw);
        for (spVec::InnerIterator it(cE); it; ++it)
        {
            double mu_rho = mu - rho*it.value()/(2*rho);
            double LS_res = mu_rho + sqrt(pow(mu_rho, 2) + mu*it.value()/(2*rho));
            w.insert(Nx + it.index()) = LS_res;
            w.insert(Nx + Nh_w + it.index()) = LS_res + it.index();
        }
    }
}

#endif