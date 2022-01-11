#ifndef FIPOPT_RESTORATION_PHASE_DENSE_HPP
#define FIPOPT_RESTORATION_PHASE_DENSE_HPP

#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Algorithms/Barrier_Subproblem/Barrier_Subproblem.hpp>
#include <Common/Barrier_Parameter_Update/Barrier_Parameter_Update.hpp>
#include <Dense/Optimality/Optimality.hpp>
#include <Dense/Algorithms/LSFB/LSFB_Status.hpp>
#include <Dense/Algorithms/LSFB/LSFB_Param.hpp>
#include <Dense/Algorithms/LSFB/LSFB_Journalist.hpp>
#include <Dense/Functors/Objective/Objective_Restoration/Equality_Restoration.hpp>
#include <Dense/Functors/Objective/Objective_Restoration/Inequality_Restoration.hpp>
#include <Dense/Restoration_Least_Squares/Restoration_Least_Squares.hpp>
#include <Dense/Initial_Multipliers/Initial_Multipliers.hpp>
#include <Common/Utils/Print.hpp>

#include <fstream>
namespace FIPOPT::Dense
{
    template <template <class> typename LinSolver, template <class, int, int, int> class Objective_Restoration,
              double (*Update_mu)(const double &) = Superlinear_mu_Update>
    struct Restoration_Solver
    {
        template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x, typename Vec_cI>
        static LSFB_status Solve_LSFB(objective<Derived, Nx, Ng, Nh> &f,
                                      MatrixBase<Vec_x> &x,
                                      MatrixBase<Vec_cI> &z,
                                      double &mu_bar,
                                      const std::string &fPath,
                                      const LSFB_param &P = LSFB_param())
        {

            Objective_Restoration f_R(f, x, mu_bar, P.rho);
            constexpr static int Nw = f_R.Nx;
            constexpr static int Nh_w = f_R.Nh;
            using Vec_w = Eigen::Matrix<double, f_R.Nx, 1>;
            using Vec_cI_w = Eigen::Matrix<double, 2 * Nw, 1>;
            using Vec_cE_w = Eigen::Matrix<double, Nh_w, 1>;
            LSFB_iteration_journalist LSFB_journalist(fPath, P.s_max);
            double tau;
            Vec_w w = Eval_Restoration_Least_Squares(f_R, mu_bar, P.rho, x);
            Vec_cI_w z_w;
            z_w.head(Nx) = (z.head(Nx).array() < P.rho).select(z.head(Nx), Vec_x::Constant(P.rho));

            z_w.segment(Nx, Nw - Nx) = mu_bar * w.tail(Nw - Nx).cwiseInverse();
            z_w = (z_w.array() > 0).select(z_w, Vec_cI_w::Constant(1e-6));
            z_w = (z_w.array() < 1e20).select(z_w, Vec_cI_w::Constant(1e20));
            Vec_cE_w lbd_w = Vec_cE_w::Constant(0.);

            for (int t = 0; t < P.iter_max; t++)
            {
                LSFB_journalist.Write(f_R, w, lbd_w, z_w, mu_bar);
                if (Eval_Global_Optimality_Error(f_R, w, lbd_w, z_w, P.s_max) < P.eps_tol)
                {
                    if (t == 0)
                    {
                        return LSFB_INFEASIBLE;
                    }
                    x = w.head(Nx);
                    return LSFB_ACCEPTED;
                }
                if (barrier_subproblem<LinSolver>::Solve(f_R, w, lbd_w, z_w, mu_bar, tau, fPath + "mu_" + std::to_string(mu_bar) + "/") == BS_INFEASIBLE)
                    break;
                    
                Update_FTTB_tau(mu_bar, tau);
                mu_bar = Update_mu(mu_bar);
            }
            x = w.head(Nx);
            w = Eval_Restoration_Least_Squares(f_R, mu_bar, P.rho, x);
            x = w.head(Nx);
            return LSFB_ACCEPTED;
        }
    };

}
#endif