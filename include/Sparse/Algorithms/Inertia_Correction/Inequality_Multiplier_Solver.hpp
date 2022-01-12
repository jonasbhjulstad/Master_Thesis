#ifndef FIPOPT_INEQUALITY_MULTIPLIER_SOLVER_HPP
#define FIPOPT_INEQUALITY_MULTIPLIER_SOLVER_HPP
#include <Common/EigenDataTypes.hpp>
namespace FIPOPT::Sparse
{
    template <typename Derived, typename dVec, typename Vec_z>
    requires is_Objective<Objective> && is_MatrixBase<dVec, Vec_z>
    inline void Solve_z_step(
        objective<Derived> &f,
        const dVec &x,
        const dVec &d_x,
        const Vec_z &z,
        Vec_z &d_z,
        const double &mu)
    {
        dVec S_inv = f.Eval_cI(x).cwiseInverse();
        dVec Sigma = f.Eval_cI(x).cwiseInverse().cwiseProduct(z);
        // d_z = z - f.Eval_cI(x).cwiseInverse()*mu + (Sigma.asDiagonal()*f.Eval_grad_cI(x)*d_x);
        // d_z = (-S_inv.cwiseProduct(z)).asDiagonal()*f.Eval_grad_cI(x)*d_x + z - mu*S_inv;
        // d_z = -Sigma.cwiseProduct(f.Eval_grad_cI(x)*d_x  + f.Eval_cI(x) + mu*z.cwiseInverse());
        d_z = -Sigma.cwiseProduct(f.Eval_grad_cI(x)*d_x) - z + mu*f.Eval_cI(x).cwiseInverse();
    }
}

#endif