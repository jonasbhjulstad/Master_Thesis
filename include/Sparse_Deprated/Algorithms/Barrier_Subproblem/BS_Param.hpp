#ifndef FIPOPT_BS_PARAM_HPP
#define FIPOPT_BS_PARAM_HPP
namespace FIPOPT::Sparse
{
    struct BS_param
    {
        BS_param(const double &mu) : iter_max(100), kappa_Sigma(1e10), kappa_eps(1.), tau_min(.99),
                                     s_max(100), lbd_max(1e3) {}

        int iter_max;
        double kappa_Sigma;
        double kappa_eps;
        double s_max;
        double tau_min;
        double lbd_max;
    };
}
#endif