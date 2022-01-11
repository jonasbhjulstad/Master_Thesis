#ifndef FIPOPT_HESSIAN_SAFEGUARDS_HPP
#define FIPOPT_HESSIAN_SAFEGUARDS_HPP
namespace FIPOPT::Dense
{
    template <typename Derived, int Nx, int Ng, int Nh, typename Vec_cI, typename Vec_x>
    inline void Hessian_Deviation_Reset(
        objective<Derived, Nx, Ng, Nh> &f,
        Vec_cI &z,
        const Vec_x &x,
        const double &mu,
        const double &kappa_Sigma)
    {
        z = z.array().min(kappa_Sigma * mu * f.Eval_cI(x).array().inverse()).max(mu / kappa_Sigma * f.Eval_cI(x).array().inverse());
    }
}
#endif