#ifndef FIPOPT_SOC_FILTER_LINESEARCH_HPP
#define FIPOPT_SOC_FILTER_LINESEARCH_HPP
#include <Common/EigenDataTypes.hpp>
#include <Sparse/Algorithms/SOC/SOC.hpp>
#include <Sparse/Algorithms/LineSearch/LS_Status.hpp>
namespace FIPOPT::Sparse
{

    template <typename Derived, typename Derived_B, typename LinSolver>
    LS_status Solve_Filter_LineSearch(
        objective<Derived> &f,
        barrier<Derived_B> &phi,
        FL_filter<Derived_B> &F,
        const SparseMatrixBase<spMat> & KKT_mat,
        const MatrixBase<dVec> &x,
        const MatrixBase<dVec> &lbd,
        const MatrixBase<dVec> &z,
        const double &max_iter,
        double &alpha,
        MatrixBase<dVec> &d_x,
        MatrixBase<dVec> &d_lbd,
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