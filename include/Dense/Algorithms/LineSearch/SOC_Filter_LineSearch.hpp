#ifndef FIPOPT_SOC_FILTER_LINESEARCH_HPP
#define FIPOPT_SOC_FILTER_LINESEARCH_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Algorithms/SOC/SOC.hpp>
#include <Dense/Algorithms/LineSearch/LS_Status.hpp>
namespace FIPOPT::Dense
{

    template <template <class> typename LinSolver>
    struct SOC_Filter_LineSearch
    {
    template <typename Derived, int Nx, int Ng, int Nh, typename Mat_A,typename Derived_B, typename Vec_x, typename Vec_cE, typename Vec_cI>
    static LS_status Solve(
        objective<Derived, Nx, Ng, Nh> &f,
        barrier<Derived_B, Nx, Ng, Nh> &phi,
        FL_filter<Derived_B, Nx, Ng, Nh> &F,
        const MatrixBase<Mat_A>& KKT_mat,
        const MatrixBase<Vec_x> &x,
        const MatrixBase<Vec_cE> &lbd,
        const MatrixBase<Vec_cI> &z,
        MatrixBase<Vec_x> &d_x,
        MatrixBase<Vec_cE> &d_lbd,
        double &alpha,
        const double &max_iter = 100)
    {
        alpha = 1.;

        if (F.Eval_theta(alpha) > F.Eval_theta(0.))
        {
            if (SOC_system<LinSolver>::Solve(f, phi, F, KKT_mat, x, lbd, z, d_x, d_lbd) == SOC_ACCEPTED)
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
    };
}
#endif