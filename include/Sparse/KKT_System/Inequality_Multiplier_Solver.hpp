#ifndef FIPOPT_INEQUALITY_MULTIPLIER_SOLVER_HPP
#define FIPOPT_INEQUALITY_MULTIPLIER_SOLVER_HPP
#include <Common/EigenDataTypes.hpp>
namespace FIPOPT::Sparse
{
    template <typename Derived>
    inline void Solve_z_step(
        objective<Derived> &f,
        const MatrixBase<dVec> &x,
        const MatrixBase<dVec> &d_x,
        const MatrixBase<dVec> &z,
        MatrixBase<dVec> &d_z,
        const double &mu)
    {
        dVec S_inv = f.Eval_cI(x).cwiseInverse();
        dVec Sigma = f.Eval_cI(x).cwiseInverse().cwiseProduct(z);
        d_z = -Sigma.cwiseProduct(f.Eval_grad_cI(x)*d_x) - z + mu*f.Eval_cI(x).cwiseInverse();
    }
}

#endif