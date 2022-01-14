#ifndef INTERPOLATION_HPP
#define INTERPOLATION_HPP

#include <cmath>
#include <algorithm>

inline double interpolate_cubic(const double &alpha_1,
                                const double &alpha_2,
                                const double &Phi_0,
                                const double &gPhi_0,
                                const double &Phi_1,
                                const double &Phi_2)
{
    double sq_alpha_1 = pow(alpha_1, 2);
    double sq_alpha_2 = pow(alpha_2, 2);
    double denom = sq_alpha_1 * sq_alpha_2 * (alpha_2 - alpha_1);

    double Phi_diff_2 = Phi_2 - Phi_0 - gPhi_0 * alpha_2;
    double Phi_diff_1 = Phi_1 - Phi_0 - gPhi_0 * alpha_1;

    double a = (sq_alpha_1 * Phi_diff_2 - sq_alpha_2 * Phi_diff_1) / denom;
    double b = (alpha_2 * sq_alpha_2 * Phi_diff_1 - alpha_1 * sq_alpha_1 * Phi_diff_2) / denom;

    double res = (-b + sqrt(b * b - 3 * a * gPhi_0)) / (3 * a);
    return res;
}

template <class ... objTypes>
inline double interpolate(objective_linesearch<objTypes ...>& phi,
                        
                          const double &alpha_0,
                          const double &alpha_1,
                          const double &c1)
{
    double Phi_0 = phi(alpha_0);
    double gPhi_0 = phi.Eval_grad(alpha_0);
    double Phi_1 = phi.Eval_grad(alpha_1);

    double denom = 2*(Phi_1 - Phi_0 - gPhi_0 * alpha_1);
    double alpha_2 = std::min(-gPhi_0 * alpha_1 * alpha_1 / denom, alpha_1);

    if (alpha_2 <= 0 || (std::abs(alpha_2 - alpha_0) < 1e-3))
        alpha_2 = alpha_0 / 2;

    double Phi_2 = phi(alpha_2);
    double gPhi_2 = phi.Eval_grad(alpha_2);

    if (Phi_2 <= (Phi_0 + c1 * alpha_2 * gPhi_2))
        return alpha_2;

    return std::min(interpolate_cubic(alpha_1, alpha_2, Phi_0, gPhi_0, Phi_1, Phi_2), alpha_0);
}
#endif