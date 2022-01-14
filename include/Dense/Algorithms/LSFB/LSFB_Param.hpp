#ifndef FIPOPT_LSFB_PARAM_HPP
#define FIPOPT_LSFB_PARAM_HPP
namespace FIPOPT::Dense
{
    struct LSFB_param
    {
        LSFB_param() : iter_max(200), lbd_max(100.), s_max(100.), eps_tol(1e-6), rho(1e3), kappa_1(1e-2), kappa_2(1e-2),
                       max_ineq_restoration(3), max_eq_restoration(3) {}
        double lbd_max, s_max;
        int iter_max;
        double eps_tol;
        double rho;
        double kappa_1, kappa_2;
        int max_ineq_restoration, max_eq_restoration;
    };
}

#endif