#ifndef FIPOPT_INEQUALITY_MULTIPLIER_SOLVER_HPP
#define FIPOPT_INEQUALITY_MULTIPLIER_SOLVER_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
namespace FIPOPT::Dense
{
    template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x, typename Vec_cI>
    inline void Solve_z_step(
        objective<Derived, Nx, Ng, Nh> &f,
        const MatrixBase<Vec_x> &x,
        const MatrixBase<Vec_x> &d_x,
        const MatrixBase<Vec_cI> &z,
        MatrixBase<Vec_cI> &d_z,
        const double &mu)
    {
        Vec_cI S_inv = f.Eval_cI(x).cwiseInverse();
        Vec_cI Sigma = f.Eval_cI(x).cwiseInverse().cwiseProduct(z);
        d_z = -Sigma.cwiseProduct(f.Eval_grad_cI(x)*d_x) - z + mu*f.Eval_cI(x).cwiseInverse();
        double a = 1;
    }
}

#endif