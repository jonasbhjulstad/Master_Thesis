#ifndef FIPOPT_RESTORATION_LSFB_ALGORITHM_HPP
#define FIPOPT_RESTORATION_LSFB_ALGORITHM_HPP

#include <Sparse/Functors/Objective/Objective.hpp>
#include <Sparse/Algorithms/Barrier_Subproblem/Barrier_Subproblem.hpp>
#include <Common/Barrier_Parameter_Update/Barrier_Parameter_Update.hpp>
#include <Sparse/Optimality/Optimality.hpp>
#include <Sparse/Algorithms/LSFB/LSFB_Status.hpp>
#include <Sparse/Algorithms/LSFB/LSFB_Param.hpp>
#include <Sparse/Functors/Objective/Objective_Restoration/Equality_Restoration.hpp>
#include <Sparse/Functors/Objective/Objective_Restoration/Inequality_Restoration.hpp>
#include <Sparse/Restoration_Least_Squares/Restoration_Least_Squares.hpp>
#include <Sparse/Initial_Multipliers/Initial_Multipliers.hpp>
#include <Common/Utils/Print.hpp>
#include <fstream>
namespace FIPOPT::Sparse
{

    template <typename Derived, typename Vec_z_w, typename LinSolver>
    static LSFB_status Solve_Restoration_LSFB(objective<Derived> &f_R,
                                              dVec &w,
                                              MatrixBase<Vec_z_w> &z_w,
                                              const double &mu_0,
                                              const std::string &fPath,
                                              LinSolver &KKT_solver,
                                              double (*Update_mu)(const double &) = Superlinear_mu_Update)
    {
        const static std::string iter_dir = "Barrier_Central_Path/";
        const LSFB_param P;
        double tau;
        double mu_bar = mu_0;

        dVec lbd_w = Eval_Initial_Multipliers(f_R, w);
        // spVec_w lbd_w = spVec_w::Constant(1.);

        CSV_iteration_journalist CSV_journalist(fPath, iter_dir);
        std::ofstream f_mu(fPath + iter_dir + "mu.csv");
        std::ofstream f_next_BS(fPath + iter_dir + "Next_Subproblem.csv");

        if (Eval_Global_Optimality_Error(f_R, w, lbd_w, z_w, P.s_max) <= P.eps_tol)
        {
            return LSFB_INFEASIBLE;
        }

        for (int t = 0; t < P.iter_max; t++)
        {
            Update_FTTB_tau(mu_bar, tau);
            CSV_journalist.Write(w, lbd_w, z_w);
            if (Solve_Barrier_Subproblem(f_R, w, lbd_w, z_w, mu_bar, tau, fPath, KKT_solver) == BS_INFEASIBLE)
            {
                Eval_Restoration_Least_Squares(f_R, mu_bar, P.rho, w);
                f_next_BS << "Barrier_Subproblem";
                f_next_BS.close();
                return LSFB_ACCEPTED;
            }

            std::cout << Eval_Global_Optimality_Error(f_R, w, lbd_w, z_w, P.s_max) << std::endl;
            if (Eval_Global_Optimality_Error(f_R, w, lbd_w, z_w, P.s_max) < P.eps_tol)
            {
                std::cout << "Solved to global tolerance" << std::endl;
                f_next_BS << "Barrier_Subproblem";
                f_next_BS.close();
                return LSFB_ACCEPTED;
            }
            f_next_BS << "Restoration_Subproblem\n";
            f_mu << mu_bar << '\n';
            mu_bar = Update_mu(mu_bar);
        }
        f_next_BS.close();
        std::cout << "Max Iteration Exceeded: Restoration_LSFB" << std::endl;
        return LSFB_MAX_ITERATION_EXCEEDED;
    }

    template <typename Derived, typename Vec_x, typename Vec_cI, typename LinSolver>
    static LSFB_status Solve_Restoration_Phase(objective<Derived> &f_R,
                                               MatrixBase<Vec_x> &x,
                                               MatrixBase<Vec_cI> &z_x,
                                               const double &mu_bar,
                                               const std::string &fPath,
                                               LinSolver &KKT_solver,
                                               const LSFB_param &P = LSFB_param())
    {
        const int Nw = f_R.Get_Nx();
        const int Nh_w = f_R.Get_Nh();
        const int Nx = Nw - 2 * Nh_w;
        dVec w = x;
        w.conservativeResize(Nw);
        Eval_Restoration_Least_Squares(f_R, mu_bar, P.rho, w);
        dVec z_w(2*Nw);
        z_w.head(Nx) = (z_x.array() < P.rho).select(z_x, Vec_x::Constant(P.rho));
        z_w.segment(Nx, Nw-Nx) = mu_bar * w.tail(Nw - Nx).cwiseInverse();
        // z_w = spVec_w::Constant(1.);
        if (Solve_Restoration_LSFB(f_R, w, z_w, mu_bar, fPath, KKT_solver) == LSFB_ACCEPTED)
        {
            x = w.topRows(Nx);
            return LSFB_ACCEPTED;
        }
        else
        {
            return LSFB_INFEASIBLE;
        }
    }
}
#endif