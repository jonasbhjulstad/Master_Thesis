#ifndef FIPOPT_BARRIER_SUBPROBLEM_HPP
#define FIPOPT_BARRIER_SUBPROBLEM_HPP
#include <Dense/KKT_System/Inequality_Multiplier_Solver.hpp>
#include <Dense/Hessian_Safeguards/Hessian_Safeguards.hpp>
#include <Dense/Algorithms/Inertia_Correction/Inertia_Correction.hpp>
#include <Dense/Algorithms/LineSearch/Fraction_To_The_Boundary/Fraction_To_The_Boundary.hpp>
#include <Dense/Functors/Filter/FL_Filter.hpp>
#include <Dense/Algorithms/LineSearch/SOC_Filter_LineSearch.hpp>
#include <Dense/Algorithms/SOC/SOC.hpp>
#include <Dense/Algorithms/Barrier_Subproblem/BS_Param.hpp>
#include <Dense/Algorithms/Barrier_Subproblem/BS_Status.hpp>
#include <Dense/Functors/Barrier/Logbarrier_Memoized.hpp>
#include <Dense/Algorithms/Barrier_Subproblem/BS_Journalist.hpp>
#include <Common/Utils/Print.hpp>
namespace FIPOPT::Dense
{
    template <template <class> typename LinSolver, template <class, int, int, int> typename Barrier = logbarrier>
    struct barrier_subproblem
    {
        template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x, typename Vec_cE, typename Vec_cI>
        static BS_status Solve(
            objective<Derived, Nx, Ng, Nh> &f,
            MatrixBase<Vec_x> &x,
            MatrixBase<Vec_cE> &lbd,
            MatrixBase<Vec_cI> &z,
            const double& mu,
            const double &tau_j,
            const std::string& fPath)
        {
            using Vec_A = Eigen::Matrix<double, Nx + Nh, 1>;
            using Mat_A = Eigen::Matrix<double, Nx + Nh, Nx + Nh>;

            BS_param P(mu);
            BS_iteration_journalist BS_journalist(fPath, P.s_max);
            Mat_A KKT_mat = Mat_A::Zero();
            Vec_A KKT_vec = Vec_A::Zero();
            Vec_x d_x = Vec_x::Zero();
            Vec_cE d_lbd = Vec_cE::Zero();
            Vec_cI d_z = Vec_cI::Zero();
            double alpha = 1;
            double alpha_z_g = 1;
            double alpha_z_ub = 1;
            double alpha_z_lb = 1;
            double delta_w_last = 0;
            FL_filter_set filter_set;
            Barrier<Derived, Nx, Ng, Nh> phi(f, mu);
            FL_filter F(phi, x, filter_set);

            for (int i = 0; i < P.iter_max; i++)
            {
                BS_journalist.Write(f, x, lbd, z, alpha, alpha_z_g, alpha_z_ub, alpha_z_lb);
                if ((Eval_Barrier_Optimality_Error(f, phi, x, lbd, z, P.s_max) <= P.kappa_eps * mu) && (f.Eval_cI(x).array() >= 0).all())
                {
                    return BS_ACCEPTED;
                }

                // Print_Iteration_States(x, lbd, z);
                Eval_KKT_Jacobian(f, KKT_mat, x, lbd, z, mu);
                Eval_KKT_Value(f, phi, KKT_vec, x, lbd, z);

                if (IC_solver<LinSolver>::Solve(KKT_mat, KKT_vec, d_x, d_lbd, delta_w_last, mu, fPath) != IC_ACCEPTED)
                {
                    break;
                }
                F.Update_Direction(x, d_x);
                if (SOC_Filter_LineSearch<LinSolver>::Solve(f, phi, F, KKT_mat,  x, lbd, z, d_x, d_lbd, alpha) == LS_INFEASIBLE)
                {
                    break;
                }

                Solve_z_step(f, x, d_x, z, d_z, mu);
                if constexpr (Ng > 0)
                    alpha_z_g = Solve_FTTB_Step(z.head(Ng), d_z.head(Ng), tau_j);
                alpha_z_ub = Solve_FTTB_Step(z.segment(Ng, Nx), d_z.segment(Ng, Nx), tau_j);
                alpha_z_lb = Solve_FTTB_Step(z.tail(Nx), d_z.tail(Nx), tau_j);
                // Print_Iteration_Steps(i, alpha, alpha_z_g, alpha_z_lb, alpha_z_ub);
                Update_PD_States(x, lbd, z, d_x, d_lbd, d_z, alpha, alpha_z_g, alpha_z_ub, alpha_z_lb);
                Hessian_Deviation_Reset(f, z, x, mu, P.kappa_Sigma);
            }

            return BS_INFEASIBLE;
        }
    };
}

#endif