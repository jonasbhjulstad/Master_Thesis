#ifndef FIPOPT_OPTIMALITY_HPP
#define FIPOPT_OPTIMALITY_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Barrier/Barrier.hpp>
#include <Dense/Functors/Barrier/Logbarrier.hpp>
#include <Eigen/Dense>
#include <iostream>
namespace FIPOPT::Dense
{
    template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x>
    inline double Eval_Primal_Infeasibility(
        objective<Derived, Nx, Ng, Nh> &f,
        const MatrixBase<Vec_x> &x)
    {
        return f.Eval_cE(x).template lpNorm<Eigen::Infinity>();
    }

    template <typename Derived, int Nx, int Ng, int Nh, typename Derived_B, typename Vec_x, typename Vec_cE, typename Vec_cI>
    inline double Eval_Dual_Infeasibility(
        objective<Derived, Nx, Ng, Nh> &f,
        barrier<Derived_B, Nx, Ng, Nh> &phi,
        const MatrixBase<Vec_x> &x,
        const MatrixBase<Vec_cE> &lbd,
        const MatrixBase<Vec_cI> &z,
        const double &s_max)
    {
        const double s_d = std::max(s_max, z.template lpNorm<1>() / (Ng + 2*Nx)) / s_max;
        Vec_x KKT_0 = -(f.Eval_grad(x) - f.Eval_grad_cE(x).transpose() * lbd - f.Eval_grad_cI(x).transpose()*z);
        return KKT_0.template lpNorm<Eigen::Infinity>() / s_d;
    }
    template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x, typename Vec_cE, typename Vec_cI>
    inline double Eval_Complementary_Slackness_Infeasibility(
        objective<Derived, Nx, Ng, Nh> &f,
        const MatrixBase<Vec_x> &x,
        const MatrixBase<Vec_cE> &lbd,
        const MatrixBase<Vec_cI> &z,
        const double& mu,
        const double &s_max)
    {
        const double s_c = std::max(s_max, (lbd.template lpNorm<1>() + z.template lpNorm<1>()) / (Nh + Ng + 2*Nx)) / s_max;
        Vec_cI KKT_2 = f.Eval_cI(x).cwiseProduct(z) - Vec_cI::Constant(mu);
        return KKT_2.template lpNorm<Eigen::Infinity>() / s_c;
    }

    template <typename Derived, int Nx, int Ng, int Nh, typename Derived_B, typename Vec_x, typename Vec_cI, typename Vec_cE>
    inline double Eval_Barrier_Optimality_Error(objective<Derived, Nx, Ng, Nh> &f,
                                                barrier<Derived_B, Nx, Ng, Nh> &phi,
                                                const MatrixBase<Vec_x> &x,
                                                const MatrixBase<Vec_cE> &lbd,
                                                const MatrixBase<Vec_cI> &z,
                                                const double &s_max)
    {
        const double inf_dual = Eval_Dual_Infeasibility(f, phi, x, lbd, z, s_max);
        const double inf_primal = Eval_Primal_Infeasibility(f, x);
        const double inf_compl = Eval_Complementary_Slackness_Infeasibility(f, x, lbd, z, phi.Get_mu(), s_max);
        return std::max({inf_dual, inf_primal, inf_compl});
    }

    template <typename Derived, int Nx, int Ng, int Nh, typename Vec_x, typename Vec_cI, typename Vec_cE>
    inline double Eval_Global_Optimality_Error(objective<Derived, Nx, Ng, Nh> &f,
                                               const MatrixBase<Vec_x> &x,
                                               const MatrixBase<Vec_cE> &lbd,
                                               const MatrixBase<Vec_cI> &z,
                                               const double &s_max)

    {
        logbarrier phi(f, 0);
        return Eval_Barrier_Optimality_Error(f, phi, x, lbd, z, s_max);
    }

}

#endif