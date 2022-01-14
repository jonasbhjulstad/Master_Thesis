#ifndef FIPOPT_BARRIER_SUBPROBLEM_HPP
#define FIPOPT_BARRIER_SUBPROBLEM_HPP
#include <Sparse/KKT_System/Inequality_Multiplier_Solver.hpp>
#include <Sparse/Hessian_Safeguards/Hessian_Safeguards.hpp>
#include <Sparse/Algorithms/Inertia_Correction/Inertia_Correction.hpp>
#include <Sparse/Algorithms/LineSearch/Fraction_To_The_Boundary/Fraction_To_The_Boundary.hpp>
#include <Sparse/Functors/Filter/FL_Filter.hpp>
#include <Sparse/Algorithms/LineSearch/SOC_Filter_LineSearch.hpp>
#include <Sparse/Algorithms/SOC/SOC.hpp>
#include <Sparse/Algorithms/Barrier_Subproblem/BS_Param.hpp>
#include <Sparse/Algorithms/Barrier_Subproblem/BS_Status.hpp>
#include <Sparse/Functors/Barrier/Logbarrier_Memoized.hpp>
#include <Common/Utils/Print.hpp>
namespace FIPOPT::Sparse
{
    template <typename Derived, typename Vec_x, typename Vec_cE, typename Vec_cI, typename LinSolver>
            BS_status Solve_Barrier_Subproblem(
                objective<Derived> &f,
                MatrixBase<Vec_x> &x,
                MatrixBase<Vec_cE> &lbd,
                MatrixBase<Vec_cI> &z,
                const double &mu,
                const double &tau_j,
                const std::string &fPath,
                LinSolver& KKT_solver)
    {
        const std::string iter_dir = "Subproblem_Iterations/mu_" + std::to_string(mu) + "/";
        const int Nx = x.rows();
        const int Nh = lbd.rows();
        const int N_A = Nx + Nh;
        const int Nz = z.rows();
        const int Ng = Nz - 2 * Nx;
        const BS_param P(mu);
        spMat KKT_mat(N_A, N_A);
        spVec KKT_vec(N_A);
        dVec d_x(Nx);
        dVec d_lbd(Nh);
        dVec d_z(Nz);

        const double max_iter = 1000;
        double alpha, alpha_z_g, alpha_z_ub, alpha_z_lb;
        logbarrier phi(f, mu);
        CSV_iteration_journalist CSV_journalist(fPath, iter_dir);
        std::ofstream f_obj(fPath + iter_dir + "obj.csv");
        FL_filter F(phi, x, d_x);

        double E_mu;
        double delta_w_last = 0;
        for (int i = 0; i < P.iter_max; i++)
        {
            E_mu = Eval_Barrier_Optimality_Error(f, phi, x, lbd, z, P.s_max);
            f_obj << E_mu << ", " << f(x) << ", " << f.Eval_cE(x).norm() << ", " << phi(x) << '\n';
            if ((E_mu <= P.kappa_eps * mu) && (all_larger(f.Eval_cI(x), 0)))
            {
                f_obj.close();
                return BS_ACCEPTED;
            }

            Print_Iteration_States(x, lbd, z);
            Eval_KKT_Jacobian(f, KKT_mat, x, lbd, z, mu);
            Eval_KKT_Value(f, phi, KKT_vec, x, lbd, z);

            if (Solve_Inertia_Correction(KKT_mat, KKT_vec, d_x, d_lbd, delta_w_last, mu, KKT_solver) != IC_ACCEPTED)
            {
                std::cout << "Infeasible IC_solve" << std::endl;
                f_obj.close();
                return BS_INFEASIBLE;
            }
            std::cout << "d_x:" << d_x.transpose() << std::endl;
            F.Update_Direction(x, d_x);
            if (Solve_Filter_LineSearch(f, phi, F, KKT_mat, x, lbd, z, max_iter, alpha, d_x, d_lbd, KKT_solver) == LS_INFEASIBLE)
            {
                f_obj.close();
                std::cout << "LineSearch Infeasible" << std::endl;
                return BS_INFEASIBLE;
            }

            Solve_z_step(f, x, d_x, z, d_z, mu);
            if (Ng > 0)
                alpha_z_g = Solve_FTTB_Step(z.topRows(Ng), d_z.topRows(Ng), tau_j);
            alpha_z_ub = Solve_FTTB_Step(z.middleRows(Ng, Nx), d_z.middleRows(Ng, Nx), tau_j);
            alpha_z_lb = Solve_FTTB_Step(z.bottomRows(Nx), d_z.bottomRows(Nx), tau_j);
            Update_PD_States(x, lbd, z, d_x, d_lbd, d_z, alpha, alpha_z_g, alpha_z_ub, alpha_z_lb);
            Hessian_Deviation_Reset(f, z, x, mu, P.kappa_Sigma);
            CSV_journalist.Write(x, lbd, z);
        }
        f_obj.close();

        return BS_MAX_ITERATION_EXCEEDED;
    }
}

#endif