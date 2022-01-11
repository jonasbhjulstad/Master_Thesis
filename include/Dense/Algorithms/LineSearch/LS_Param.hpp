#ifndef FIPOPT_LS_PARAM_HPP
#define FIPOPT_LS_PARAM_HPP
namespace FIPOPT::Dense
{
    struct LS_param
    {
        const double rho, c1, c2;
        const uint32_t zoom_iter_max, LS_iter_max;
    };
}

#endif