#ifndef FIPOPT_IC_SOLVER_HPP
#define FIPOPT_IC_SOLVER_HPP
#include <Common/EigenDataTypes.hpp>
#include <Common/Utils/Eigen_Utils.hpp>
#include <Eigen/Eigenvalues>
#include <Sparse/Algorithms/Inertia_Correction/IC_Param.hpp>
#include <Sparse/Algorithms/Inertia_Correction/IC_Status.hpp>
#include <Sparse/KKT_System/KKT_System.hpp>
namespace FIPOPT::Sparse
{
    template <typename T>
    inline bool Eval_Inertia_Condition(const SparseMatrixBase<T> &KKT_mat, const double &zero_tol, const int& Nx, const int& Nh)
    {   
        Eigen::ArrayXd eigvals = dVec(KKT_mat).eigenvalues().array().real();
        return (((eigvals > zero_tol).count() == Nx) && ((eigvals < 0).count() == Nh)); 
    }

    template <typename Vec_A>
    inline void Update_deltas(const MatrixBase<Vec_A> &KKT_eigvals,
                              double &delta_c,
                              double &delta_w,
                              const double &delta_w_last,
                              const double &mu,
                              const IC_param &P)
    {
        delta_c = ((KKT_eigvals.array().abs() < P.zero_tol).any()) ? P.delta_c_bar * pow(mu, P.kappa_c) : 0;
        delta_w = (delta_w_last == 0) ? P.delta_w_0 : std::max(P.delta_w_min, P.kappa_w_n * delta_w_last);
    }

        template <typename LinSolver>
        IC_status Solve_Inertia_Correction(
            SparseMatrixBase<spMat> &KKT_mat,
            SparseMatrixBase<spVec> &KKT_vec,
            spVec &d_x,
            spVec &d_lbd,
            double &delta_w_last,
            const double &mu,
            LinSolver& KKT_solver,
            const IC_param &P = IC_param())
        {
            const int Nx = d_x.rows();
            const int Nh = d_lbd.rows();
            const int N_A = Nx + Nh;

            KKT_solver.compute(KKT_mat);
            double delta_c, delta_w;
            if (Eval_Inertia_Condition(KKT_mat, P.zero_tol, Nx, Nh) && Solve_KKT_System(KKT_vec, d_x, d_lbd, KKT_solver))
                return IC_ACCEPTED;

            spMat delta_mat(N_A, N_A);
            while(delta_w < P.delta_w_max)
            {
                spVec sol(N_A); 
                dVec KKT_eigvals(N_A);
                KKT_eigvals = dVec(KKT_mat + delta_mat).eigenvalues().real();
                Update_deltas(KKT_eigvals, delta_c, delta_w, delta_w_last, mu, P);
                set_diagonal(delta_mat, delta_w, 0, 0, Nx);
                set_diagonal(delta_mat, -delta_c, Nx, Nx, Nh);
                KKT_solver.compute(KKT_mat + delta_mat);
                if (Eval_Inertia_Condition(KKT_mat + delta_mat, P.zero_tol, Nx, Nh) && Solve_KKT_System(KKT_vec, d_x, d_lbd, KKT_solver))
                {
                    KKT_mat += delta_mat;
                    delta_w_last = delta_w;
                    std::cout << "Inertia corrected\t delta_w:\t" << delta_w << "\t delta_c:\t" << delta_c << std::endl;
                    return IC_ACCEPTED;
                }
                delta_w_last = (delta_w_last == 0) ? P.kappa_w_p_bar * delta_w : P.kappa_w_p * delta_w;

            }
            return IC_INFEASIBLE;
        }


}
#endif