#ifndef FIPOPT_SOC_FILTER_LINESEARCH_HPP
#define FIPOPT_SOC_FILTER_LINESEARCH_HPP
#include <Common/EigenDataTypes.hpp>
#include <Sparse/Algorithms/SOC/SOC.hpp>
#include <Sparse/Algorithms/LineSearch/LS_Status.hpp>
namespace FIPOPT::Sparse
{

    template <typename Derived, typename Derived_B, 
    typename Mat_A, typename Vec_x, typename Vec_cE, typename Vec_cI,
    typename LinSolver>
    LS_status Solve_Filter_LineSearch(
        objective<Derived> &f,
        barrier<Derived_B> &phi,
        FL_filter<Derived_B> &F,
        const SparseMatrixBase<Mat_A> & KKT_mat,
        const SparseMatrixBase<Vec_x> &x,
        const SparseMatrixBase<Vec_cE> &lbd,
        const SparseMatrixBase<Vec_cI> &z,
        const double &max_iter,
        double &alpha,
        SparseMatrixBase<Vec_x> &d_x,
        SparseMatrixBase<Vec_cE> &d_lbd,
        LinSolver& KKT_solver)
    {
        alpha = 1.;

        if (F.Eval_theta(alpha) < F.Eval_theta(0.))
        {
            if (Solve_SOC_System(f, phi, F, KKT_mat, x, lbd, z, d_x, d_lbd, KKT_solver) == SOC_ACCEPTED)
                return LS_ACCEPTED;
        }

        for (int l = 0; l < max_iter; l++)
        {
            switch (F.Eval_Update_Filter(alpha))
            {
            case FILTER_INFEASIBLE_STEP_SIZE:
                return LS_INFEASIBLE;
                break;
            case FILTER_ACCEPTED:
                return LS_ACCEPTED;
                break;
            case FILTER_REJECTED:
                alpha /= 2;
                break;
            }
        }
        return LS_MAX_ITERATIONS_EXCEEDED;
    }
}
#endif