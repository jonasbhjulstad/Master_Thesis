#ifndef FIPOPT_HESSIAN_SAFEGUARDS_HPP
#define FIPOPT_HESSIAN_SAFEGUARDS_HPP
namespace FIPOPT::Sparse
{
    template <typename Derived, typename Vec_cI, typename Vec_x>
    inline void Hessian_Deviation_Reset(
        objective<Derived> &f,
        Vec_cI &z,
        const Vec_x &x,
        const double &mu,
        const double &kappa_Sigma)
    {
        z = z.cwiseMin(kappa_Sigma * mu * f.Eval_cI(x).cwiseInverse()).cwiseMax(mu / kappa_Sigma * f.Eval_cI(x).cwiseInverse());
    }
}
#endif