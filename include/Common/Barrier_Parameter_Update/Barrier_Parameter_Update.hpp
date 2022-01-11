#ifndef FIPOPT_BARRIER_PARAMETER_UPDATE_HPP
#define FIPOPT_BARRIER_PARAMETER_UPDATE_HPP
#include <algorithm>

namespace FIPOPT
{
    struct superlinear_mu_param
    {
        superlinear_mu_param() : eps_tol_(1e-8), kappa_eps_(10.), theta_mu_(1.5), kappa_mu_(0.2) {}
        const double eps_tol_, kappa_eps_, theta_mu_, kappa_mu_;
    };


    inline double Superlinear_mu_Update(const double &mu, const superlinear_mu_param &P)
    {
        return std::max(P.eps_tol_, std::min(P.kappa_mu_ * mu, pow(mu, P.theta_mu_)));
    }

    inline double Superlinear_mu_Update(const double &mu)
    {
        return Superlinear_mu_Update(mu, superlinear_mu_param());
    }
    
}

#endif
