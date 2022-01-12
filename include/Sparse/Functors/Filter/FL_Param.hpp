#ifndef FIPOPT_FL_PARAM_HPP
#define FIPOPT_FL_PARAM_HPP
#include <vector>
namespace FIPOPT::Sparse
{
    struct FL_condition_pair
    {
        double theta = 1e20;
        double phi = 1e20;
    };
    typedef std::vector<FL_condition_pair> FL_filter_set;

    struct FL_param
    {
        FL_param(const double &theta_min_, const double &theta_max_) : gamma_theta(1e-5), gamma_phi(1e-5), delta(1.), gamma_alpha(0.05),
                                                                       s_theta(2.3), s_phi(1.1), eta(1e-4), theta_max(theta_max_), theta_min(theta_min_), filter_set_size(100) {}

        double gamma_theta, gamma_phi, gamma_alpha;
        double delta;
        double eta;
        double theta_min, theta_max;
        double s_theta, s_phi;
        int filter_set_size;
    };

}
#endif