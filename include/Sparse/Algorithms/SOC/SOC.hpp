#ifndef FIPOPT_SOC_Sparse_HPP
#define FIPOPT_SOC_Sparse_HPP
#include <Sparse/Functors/Objective/Objective.hpp>
#include <Sparse/Functors/Barrier/Barrier.hpp>
#include <Sparse/Functors/Filter/Filter_Status.hpp>
#include <Sparse/Functors/Filter/FL_Filter.hpp>
namespace FIPOPT::Sparse
{
    enum SOC_status
    {
        SOC_INFEASIBLE,
        SOC_REJECTED,
        SOC_INSUFFICIENT_REDUCTION,
        SOC_MAX_ITERATION_EXCEEDED,
        SOC_ACCEPTED
    };
    struct SOC_param
    {
        SOC_param() : kappa_soc(.99), max_iter(4) {}
        double kappa_soc;
        int max_iter;
    };

    template <typename Derived, typename Derived_B, typename dVec, typename dVec, typename dVec>
    inline void Eval_KKT_Value_SOC(
        objective<Derived> &f,
        barrier<Derived_B> &phi,
        const MatrixBase<dVec> &x,
        const MatrixBase<dVec> &d_x,
        const MatrixBase<dVec> &lbd,
        const MatrixBase<dVec> &z,
        const double &alpha,
        dVec &KKT_vec)
    {
        const int Nx = x.rows();
        const int Nh = lbd.rows();

        KKT_vec.head(Nx) = -(phi.Eval_grad(x) - f.Eval_grad_cE(x).transpose() * lbd);
    }

    template <typename Derived, typename Derived_B,typename LinSolver>
    SOC_status Solve_SOC_System(objective<Derived> &f,
                                barrier<Derived_B> &phi,
                                FL_filter<Derived_B> &F,
                                const SparseMatrixBase<spMat> &KKT_mat,
                                const MatrixBase<dVec> &x,
                                const MatrixBase<dVec> &lbd,
                                const MatrixBase<dVec> &z,
                                MatrixBase<dVec> &d_x,
                                MatrixBase<dVec> &d_lbd,
                                LinSolver &KKT_solver,
                                const double &alpha = 1.,
                                const SOC_param &P = SOC_param())
    {
        const int N_A = KKT_mat.rows();
        const int Nx = x.rows();
        const int Nh = lbd.rows();
        KKT_solver.compute(KKT_mat);
        dVec KKT_vec_soc(N_A), sol(N_A);
        dVec x_soc(Nx);
        dVec d_x_cor = d_x;
        double theta_old = F.Eval_theta(0.);

        F.Update_Direction(x, d_x);
        if (F.Eval_Switching_Condition(1.) && F.Eval_Armijo_Condition(1.))
        {
            for (int i = 0; i < P.max_iter; i++)
            {
                Eval_KKT_Value_SOC(f, phi, x, d_x_cor, lbd, z, alpha, KKT_vec_soc);
                sol = KKT_solver.solve(KKT_vec_soc);
                d_x_cor = sol.head(Nx);
                F.Update_Direction(x, d_x_cor);
                switch (F.Eval_Armijo_Condition(1.))
                {
                case FILTER_ACCEPTED:
                    d_x = sol.head(Nx);
                    d_lbd = sol.tail(Nh);
                    return SOC_ACCEPTED;
                    break;
                case FILTER_INFEASIBLE_STEP_SIZE:
                    return SOC_INFEASIBLE;
                default:
                    break;
                }

                if (theta_old - F.Eval_theta(1.) > P.kappa_soc)
                {
                    return SOC_INSUFFICIENT_REDUCTION;
                }
            }
        }
        return SOC_MAX_ITERATION_EXCEEDED;
    }
}

#endif