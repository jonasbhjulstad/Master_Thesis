#ifndef FIPOPT_IC_PARAM_HPP
#define FIPOPT_IC_PARAM_HPP

namespace FIPOPT::Sparse
{
    struct IC_param
    {
        IC_param() : delta_w_min(1e-20), delta_w_0(1e-4), delta_w_max(1e40),
                     delta_c_bar(1e-8), kappa_w_n(1. / 3), kappa_w_p(8.), kappa_w_p_bar(100.), kappa_c(1. / 4), iter_max(500), zero_tol(1e-6) {}

        double delta_w_min, delta_w_0, delta_w_max, delta_c_bar;
        double kappa_w_n, kappa_w_p, kappa_w_p_bar, kappa_c, zero_tol;
        int iter_max;
    };
}

#endif