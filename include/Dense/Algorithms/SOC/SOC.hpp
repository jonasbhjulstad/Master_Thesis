#ifndef FIPOPT_SOC_DENSE_HPP
#define FIPOPT_SOC_DENSE_HPP
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Barrier/Barrier.hpp>
#include <Dense/Functors/Filter/Filter_Status.hpp>
#include <Dense/Functors/Filter/FL_Filter.hpp>
namespace FIPOPT::Dense
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

    template <typename Vec_A, typename Vec_x, typename Vec_cE, typename Vec_cI, typename Derived, int Nx, int Ng, int Nh, typename Derived_B>
    inline void Eval_KKT_Value_SOC(
        objective<Derived, Nx, Ng, Nh> &f,
        barrier<Derived_B, Nx, Ng, Nh> &phi,
        const MatrixBase<Vec_x> &x,
        const MatrixBase<Vec_x> &d_x,
        const MatrixBase<Vec_cE> &lbd,
        const MatrixBase<Vec_cI> &z,
        const double &alpha,
        Vec_A &KKT_vec)
    {
        KKT_vec.head(Nx) = -(phi.Eval_grad(x) - f.Eval_grad_cE(x).transpose() * lbd);
        KKT_vec.tail(Nh) = -(f.Eval_cE(x) +f.Eval_cE(Vec_x(x + alpha * d_x)));
    }

    template <template <class> typename LinSolver>
    struct SOC_system
    {
        template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x, typename Vec_cE, typename Vec_cI, typename Mat_A, typename Derived_B>
        static SOC_status Solve(objective<Derived, Nx, Ng, Nh> &f,
                                barrier<Derived_B, Nx, Ng, Nh> &phi,
                                FL_filter<Derived_B, Nx, Ng, Nh> &F,
                                const MatrixBase<Mat_A> &KKT_mat,
                                const MatrixBase<Vec_x> &x,
                                const MatrixBase<Vec_cE> &lbd,
                                const MatrixBase<Vec_cI> &z,
                                MatrixBase<Vec_x> &d_x,
                                MatrixBase<Vec_cE> &d_lbd,
                                const double &alpha = 1.,
                                const SOC_param &P = SOC_param())
        {
            using Vec_A = Eigen::Matrix<double, Nx + Nh, 1>;
            LinSolver<Mat_A> KKT_solver(KKT_mat);
            double theta_old = F.Eval_theta(0.);
            F.Update_Direction(x, d_x);
            Vec_x d_x_cor = d_x;
            Vec_x x_soc;
            Vec_A KKT_vec_soc, sol;
                for (int i = 0; i < P.max_iter; i++)
                {
                    Eval_KKT_Value_SOC(f, phi, x, d_x, lbd, z, alpha, KKT_vec_soc);
                    sol = KKT_solver.solve(KKT_vec_soc);
                    d_x_cor = sol.head(Nx);
                    F.Update_Direction(x, d_x_cor);
                    if (theta_old - F.Eval_theta(1.) > P.kappa_soc)
                    {
                        F.Update_Direction(x, d_x);
                        return SOC_INSUFFICIENT_REDUCTION;
                    }
                    if (F.Eval_Switching_Condition(1.) && F.Eval_Armijo_Condition(1.))
                    {
                        // std::cout << "SOC Accepted!" << std::endl;
                        d_x = sol.head(Nx);
                        d_lbd = -sol.tail(Nh);
                        return SOC_ACCEPTED;
                    }

                }
            F.Update_Direction(x, d_x);
            return SOC_MAX_ITERATION_EXCEEDED;
        }
    };
}

#endif