#ifndef FIPOPT_LSFB_ALGORITHM_HPP
#define FIPOPT_LSFB_ALGORITHM_HPP

#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Algorithms/Barrier_Subproblem/Barrier_Subproblem.hpp>
#include <Dense/Algorithms/LSFB/LSFB_Status.hpp>
#include <Dense/Algorithms/LSFB/LSFB_Param.hpp>
#include <Dense/Algorithms/Restoration_Phase/Restoration_Phase.hpp>
#include <Dense/Functors/Objective/Objective_Restoration/Inequality_Restoration.hpp>
#include <Dense/Functors/Objective/Objective_Restoration/Equality_Restoration.hpp>
#include <Dense/Functors/Objective/Objective_Messenger/Objective_Messenger.hpp>
#include <Dense/Initial_Multipliers/Initial_Multipliers.hpp>
#include <Dense/Optimality/Optimality.hpp>
#include <Dense/Algorithms/LSFB/LSFB_Journalist.hpp>
#include <Common/Utils/Print.hpp>
#include <filesystem>
#include <fstream>
namespace FIPOPT::Dense
{

    template <template <class> typename LinSolver,
              double (*Update_mu)(const double &) = Superlinear_mu_Update>
    struct LSFB
    {

        template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x>
        static LSFB_status Solve(objective<Derived, Nx, Ng, Nh> &f,
                                 MatrixBase<Vec_x> &x,
                                 const std::string &fPath,
                                 LSFB_param P = LSFB_param(),
                                 double mu = 0.1)
        {

            using Vec_cE = Eigen::Matrix<double, Nh, 1>;
            using Vec_A = Eigen::Matrix<double, Nx + Nh, 1>;
            using Mat_A = Eigen::Matrix<double, Nx + Nh, Nx + Nh>;
            using Vec_cI = Eigen::Matrix<double, Ng + 2 * Nx, 1>;
            LSFB_iteration_journalist LSFB_journalist(fPath, P.s_max);
            double tau = 0;
            int N_ineq_restoration = 0;
            int N_eq_restoration = 0;
            x = Eval_Bounded_Initial(x, f.Get_x_lb(), f.Get_x_ub(), P.kappa_1, P.kappa_2);
            Vec_cI z = Eval_Initial_Inequality_Multipliers(f, x, mu);
            Vec_cE lbd = Eval_Initial_Equality_Multipliers(f, x, Vec_x(z.segment(Nh, Nx)), Vec_x(z.tail(Nx)), P.lbd_max);
            for (int j = 0; j < P.iter_max; j++)
            {
                LSFB_journalist.Write(f, x, lbd, z, mu);
                if (Eval_Global_Optimality_Error(f, x, lbd, z, P.s_max) < P.eps_tol)
                {
                    return LSFB_ACCEPTED;
                }

                if (barrier_subproblem<LinSolver>::Solve(f, x, lbd, z, mu, tau, fPath + "mu_" + std::to_string(mu) + "/") == BS_INFEASIBLE)
                {
                    if ((N_eq_restoration >= P.max_eq_restoration) || (N_ineq_restoration >= P.max_ineq_restoration))
                    {
                        return LSFB_INFEASIBLE;
                    }
                    if (((f.Eval_cI(x).array() < 0).count()) && (Ng > 0))
                    {
                        double mu_bar = std::max(mu, f.Eval_cI(x).head(Ng).norm());
                        if (Restoration_Solver<LinSolver, inequality_restoration>::Solve_LSFB(f, x, z, mu_bar, fPath + "Inequality_Restoration_" + std::to_string(j) + "/", P) == LSFB_INFEASIBLE)
                            break;
                        N_ineq_restoration++;
                    }
                    else if (Nh > 0)
                    {
                        double mu_bar = std::max(mu, f.Eval_cE(x).norm());
                        if (Restoration_Solver<LinSolver, equality_restoration>::Solve_LSFB(f, x, z, mu_bar, fPath + "Equality_Restoration_" + std::to_string(j) + "/", P) == LSFB_INFEASIBLE)
                            break;
                        N_eq_restoration++;
                    }
                    z = Eval_Initial_Inequality_Multipliers(f, x, mu);
                    lbd = Eval_Initial_Equality_Multipliers(f, x, Vec_x(z.segment(Nh, Nx)), Vec_x(z.tail(Nx)), P.lbd_max);
                }
                Update_FTTB_tau(mu, tau);
                mu = Update_mu(mu);
            }
            LSFB_journalist.Write(f, x, lbd, z, mu);
            return LSFB_INFEASIBLE;
        }
    };
}
#endif