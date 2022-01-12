#ifndef FIPOPT_HESSIAN_SAFEGUARDS_HPP
#define FIPOPT_HESSIAN_SAFEGUARDS_HPP
namespace FIPOPT::Sparse
{
    template <typename Derived>
    inline void Hessian_Deviation_Reset(
        objective<Derived> &f,
        MatrixBase<dVec> &z,
        const MatrixBase<dVec> &x,
        const double &mu,
        const double &kappa_Sigma)
    {
        dVec correction_values = (kappa_Sigma * mu * f.Eval_cI(x).cwiseInverse()).cwiseMax(mu / kappa_Sigma * f.Eval_cI(x).cwiseInverse());
        z = (z.array() < correction_values.array()).select(z, correction_values);
    }
}
#endif