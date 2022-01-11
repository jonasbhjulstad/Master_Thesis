#ifndef LINESEARCH_STATIC_HPP
#define LINESEARCH_STATIC_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Barrier/Barrier_LineSearch/Barrier_LineSearch.hpp>
#include <Interpolation/Interpolation.hpp>
#include <LS_Param.hpp>
#include <LS_Status.hpp>
#include <algorithm>

namespace FIPOPT::Dense
{

    template <typename Obj_LineSearch>
    inline double Zoom(
        Obj_LineSearch &phi,
        double &alpha_lo,
        double &alpha_hi,
        const double &Phi_0,
        const double &grad_Phi_0,
        const double &c1,
        const double &c2,
        const double &zoom_iter_max)
    {
        for (int k = 0; k < zoom_iter_max; k++)
        {
            const double alpha_trial = interpolate(phi, alpha_lo, alpha_hi, c1);
            const double Phi_trial = phi(alpha_trial);

            if (std::abs(alpha_hi - alpha_lo) < 1e-8)
                return alpha_trial;
            if ((Phi_trial > (Phi_0 + c1 * alpha_trial * grad_Phi_0)) || (phi(alpha_trial) >= phi(alpha_lo)))
            {
                alpha_hi = alpha_trial;
            }
            else
            {
                double grad_Phi_trial = phi.Eval_grad(alpha_trial);
                if (std::abs(grad_Phi_trial) <= -c2 * grad_Phi_0)
                    return alpha_trial;
                if (grad_Phi_trial * (alpha_hi - alpha_lo) >= 0)
                {
                    alpha_hi = alpha_lo;
                }
                alpha_lo = alpha_trial;
            }
        }
        return -std::numeric_limits<double>::infinity();
    }

    template <typename Derived, typename Vec>
    inline double Eval_Backtracking_LineSearch(
        Objective &f,
        const Vec &x,
        const Vec &d,
        const LS_param &P)
    {

        objective_linesearch phi(f, x, d);

        double alpha_trial = 1.;
        double alpha_old = 0.;
        double Phi_0 = phi(0.);
        double grad_Phi_0 = phi.Eval_grad(0.);
        double Phi_trial_old = Phi_0;

        for (int i = 0; i < P.LS_iter_max; i++)
        {
            double Phi_trial = phi(alpha_trial);
            if (i > 0)
            {
                if ((Phi_trial > (Phi_0 + P.c1 * alpha_trial * grad_Phi_0)) || ((Phi_trial > Phi_trial_old)))
                {
                    return Zoom(phi, alpha_old, alpha_trial, Phi_0, grad_Phi_0, P.c1, P.c2, P.zoom_iter_max);
                }
            }
            double grad_Phi_trial = phi.Eval_grad(alpha_trial);

            if (std::abs(grad_Phi_trial) <= (-P.c2 * grad_Phi_0))
                return alpha_trial;

            if (grad_Phi_trial >= 0)
                return Zoom(phi, alpha_trial, alpha_old, Phi_0, grad_Phi_0, P.c1, P.c2, P.zoom_iter_max);

            alpha_old = alpha_trial;
            alpha_trial = std::min(alpha_trial * 1.1, 1.0);
        }
        return -std::numeric_limits<double>::infinity();
    }
}
#endif