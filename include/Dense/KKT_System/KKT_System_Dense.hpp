#ifndef FIPOPT_KKT_SYSTEM_HPP
#define FIPOPT_KKT_SYSTEM_HPP

#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Barrier/Barrier.hpp>
#include <Common/EigenDataTypes.hpp>
#include <Dense/Optimality/Optimality.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <type_traits>
#include <fstream>
namespace FIPOPT::Dense
{
    template <typename Mat_A, typename Vec_x, typename Vec_cE, typename Vec_cI, typename Derived, int Nx, int Ng, int Nh>
    inline void Eval_KKT_Jacobian(
        objective<Derived, Nx, Ng, Nh> &f,
        MatrixBase<Mat_A> &KKT_mat,
        const MatrixBase<Vec_x> &x,
        const MatrixBase<Vec_cE> &lbd,
        const MatrixBase<Vec_cI> &z,
        const double &mu)
    {
        using Mat_x = Eigen::Matrix<double, Vec_x::RowsAtCompileTime, Vec_x::RowsAtCompileTime>;
        using Mat_cI = Eigen::Matrix<double, Vec_cI::RowsAtCompileTime, Vec_x::RowsAtCompileTime>;

        Mat_cI grad_cI = f.Eval_grad_cI(x);
        Vec_cI Sigma = f.Eval_cI(x).cwiseInverse().transpose() * z.asDiagonal();
    
        KKT_mat.topLeftCorner(Nx, Nx) = f.Eval_hessian(x, lbd) + grad_cI.transpose() * Sigma.asDiagonal() * grad_cI;

        //[..       grad_c]
        //[grad_c^T    ..]

        KKT_mat.bottomLeftCorner(Nh, Nx) = f.Eval_grad_cE(x);

        KKT_mat.topRightCorner(Nx, Nh) = f.Eval_grad_cE(x).transpose(); // KKT_mat.bottomLeftCorner(Nh, Nx).transpose();
    }

    template <typename Vec_A, typename Vec_x, typename Vec_cE, typename Vec_cI, typename Derived, int Nx, int Ng, int Nh, typename Derived_B>
    inline void Eval_KKT_Value(
        objective<Derived, Nx, Ng, Nh> &f,
        barrier<Derived_B, Nx, Ng, Nh> &phi,
        Vec_A &KKT_vec,
        const MatrixBase<Vec_x> &x,
        const MatrixBase<Vec_cE> &lbd,
        const MatrixBase<Vec_cI> &z)
    {
        KKT_vec.head(Nx) = -(phi.Eval_grad(x) - f.Eval_grad_cE(x).transpose() * lbd);
        KKT_vec.tail(Nh) = -f.Eval_cE(x);
    }

    template <typename Vec_A, typename Vec_x, typename Vec_cE, typename LinSolver>
    inline bool Solve_KKT_System(const MatrixBase<Vec_A> &KKT_vec,
                                 MatrixBase<Vec_x> &d_x,
                                 MatrixBase<Vec_cE> &d_lbd,
                                 SolverBase<LinSolver> &KKT_solver)
    {
        constexpr static int Nx = Vec_x::RowsAtCompileTime;
        constexpr static int Ng = Vec_cE::RowsAtCompileTime;

        Vec_A sol = KKT_solver.solve(KKT_vec);
        if (!sol.array().isNaN().any())
        {
            d_x = sol.head(Nx);
            d_lbd = -sol.tail(Ng);
            return true;
        }
        return false;
    }

    template <typename Vec_x, typename Vec_cE, typename Vec_cI>
    inline void Update_PD_States(
        MatrixBase<Vec_x> &x,
        MatrixBase<Vec_cE> &lbd,
        MatrixBase<Vec_cI> &z,
        MatrixBase<Vec_x> &d_x,
        MatrixBase<Vec_cE> &d_lbd,
        MatrixBase<Vec_cI> &d_z,
        const double &alpha,
        const double alpha_z_g,
        const double alpha_z_ub,
        const double alpha_z_lb)
    {
        constexpr static int Nx = Vec_x::RowsAtCompileTime;
        constexpr static int Nh = Vec_cE::RowsAtCompileTime;
        constexpr static int Ng = Vec_cI::RowsAtCompileTime - 2 * Nx;
        x += alpha * d_x;
        lbd += alpha * d_lbd;
        z.head(Ng) += alpha_z_g * d_z.head(Ng);
        z.segment(Ng, Nx) += alpha_z_ub * d_z.segment(Ng, Nx);
        z.tail(Nx) += alpha_z_lb * d_z.tail(Nx);
        z = (z.array() < 0).select(Vec_cI::Zero(), z);

        double a = 1;
    }

}
#endif