#ifndef FIPOPT_LSFB_ALGORITHM_SPARSE_HPP
#define FIPOPT_LSFB_ALGORITHM_SPARSE_HPP

#include <Sparse/Functors/Objective/Objective.hpp>
#include <Sparse/Algorithms/Barrier_Subproblem/Barrier_Subproblem.hpp>
#include <Sparse/Algorithms/LSFB/LSFB_Status.hpp>
#include <Sparse/Algorithms/LSFB/LSFB_Param.hpp>
#include <Sparse/Algorithms/Restoration_Phase/Restoration_Phase.hpp>
#include <Sparse/Functors/Objective/Objective_Restoration/Inequality_Restoration.hpp>
#include <Sparse/Functors/Objective/Objective_Restoration/Equality_Restoration.hpp>
#include <Sparse/Functors/Objective/Objective_Messenger/Objective_Messenger.hpp>
#include <Sparse/Initial_Multipliers/Initial_Multipliers.hpp>
#include <Sparse/Optimality/Optimality.hpp>
#include <Common/Utils/Print.hpp>
#include <filesystem>
#include <fstream>
namespace FIPOPT::Sparse
{

        template <typename Derived, typename LinSolver, typename T>
        LSFB_status Solve_LSFB(objective<Derived> &f,
                                 MatrixBase<T> &x,
                                 LinSolver& KKT_solver,
                                 const std::string &fPath,
                                 double (*Update_mu)(const double&) = Superlinear_mu_Update)
        {
            const int Nx = f.Get_Nx();
            const int Ng = f.Get_Ng();
            const int Nh = f.Get_Nh();
            const static std::string iter_dir = "Barrier_Central_Path/";
            std::cout << x.transpose() << std::endl;
            const LSFB_param P;
            double mu = 0.1;
            double tau;
            x = Eval_Bounded_Initial(x, f.Get_x_lb(), f.Get_x_ub(), P.kappa_1, P.kappa_2);
            dVec z = f.Eval_cI(x);
            z = z.cwiseInverse()*mu;
            z = (z.array() > 0).select(z, dVec::Constant(z.rows(), 1e-6));
            z = (z.array() < 1e20).select(z, dVec::Constant(z.rows(), 1e20));
            

            dVec lbd = Eval_Initial_Equality_Multipliers(f, x, dVec(z.middleRows(Nh, Nx)), dVec(z.bottomRows(Nx)), P.lbd_max);
            CSV_iteration_journalist CSV_journalist(fPath, iter_dir);
            std::ofstream f_mu(fPath + iter_dir + "mu.csv");
            std::ofstream f_obj(fPath + iter_dir + "obj.csv");
            std::ofstream f_theta(fPath + iter_dir + "theta.csv");
            std::ofstream f_next_BS(fPath + iter_dir + "Next_Subproblem.csv");
            for (int j = 0; j < P.iter_max; j++)
            {
                CSV_journalist.Write(x, lbd, z);
                std::cout << Eval_Global_Optimality_Error(f, x, lbd, z, P.s_max) << std::endl;
                f_theta << f.Eval_cE(x).template lpNorm<2>() << '\n';
                f_obj << Eval_Global_Optimality_Error(f, x, lbd, z, P.s_max) << ", " << f(x) << '\n';
                if (Eval_Global_Optimality_Error(f, x, lbd, z, P.s_max) < P.eps_tol)
                {
                    std::cout << "Solved to global tolerance" << std::endl;
                    f_obj.close();
                    f_mu.close();
                    f_theta.close();
                    std::ofstream N_iter(fPath + "N_Subproblems.csv");
                    N_iter << j;
                    N_iter.close();
                    return LSFB_ACCEPTED;
                }
                Update_FTTB_tau(mu, tau);

                if (Solve_Barrier_Subproblem(f, x, lbd, z, mu, tau, fPath, KKT_solver) == BS_INFEASIBLE)
                {
                    double mu_bar;
                    namespace fs = std::filesystem;
                    fs::current_path(fPath);
                    dVec z_x = z.head(Nx);
                    if (any_smaller(f.Eval_cI(x), 0.))
                    {
                        std::string res_fPath = fPath + "Inequality_Restoration_" + std::to_string(j) + "/";
                        mu_bar = std::max(mu, f.Eval_cI(x).topRows(Ng).norm());
                        inequality_restoration f_R(f, x, mu_bar, P.rho);
                        fs::create_directory(res_fPath);
                        Solve_Restoration_Phase(f_R, x, z_x, mu_bar, res_fPath, KKT_solver);
                        f_next_BS << "Inequality_Restoration\n";
                    }
                    else
                    {
                        std::string res_fPath = fPath + "Equality_Restoration_" + std::to_string(j) + "/";
                        mu_bar = std::max(mu, f.Eval_cE(x).norm());
                        equality_restoration f_R(f, x, mu_bar, P.rho);
                        fs::create_directory(res_fPath);
                        Solve_Restoration_Phase(f_R, x, z_x, mu_bar, res_fPath, KKT_solver);
                        f_next_BS << "Equality_Restoration\n";
                    }
                }
                f_mu << mu << '\n';
                f_next_BS << "Barrier_Subproblem\n";
                mu = Update_mu(mu);
            }
            f_mu.close();
            f_theta.close();
            std::cout << "Max Iteration Exceeded: LSFB" << std::endl;
            return LSFB_MAX_ITERATION_EXCEEDED;
        }
}
#endif