#ifndef FIPOPT_OPTIMALITY_HPP
#define FIPOPT_OPTIMALITY_HPP
#include <Common/EigenDataTypes.hpp>
#include <Sparse/Functors/Objective/Objective.hpp>
#include <Sparse/Functors/Barrier/Barrier.hpp>
#include <Sparse/Functors/Barrier/Logbarrier.hpp>
#include <Common/Utils/Eigen_Utils.hpp>
#include <iostream>
namespace FIPOPT::Sparse
{
    template <typename Derived, typename Vec_x>
    inline double Eval_Primal_Infeasibility(
        objective<Derived> &f,
        const MatrixBase<Vec_x> &x)
    {
        return linf_norm(f.Eval_cE(x));
    }

    template <typename Derived, typename Derived_B, typename Vec_x, typename Vec_cE, typename Vec_cI>
    inline double Eval_Dual_Infeasibility(
        objective<Derived> &f,
        barrier<Derived_B> &phi,
        const MatrixBase<Vec_x> &x,
        const MatrixBase<Vec_cE> &lbd,
        const MatrixBase<Vec_cI> &z,
        const double &s_max)
    {
        const double s_d = std::max(s_max, FIPOPT::l1_norm(z) / z.rows()) / s_max;
        spMat KKT_0 = -(f.Eval_grad(x) - f.Eval_grad_cE(x).transpose() * lbd - f.Eval_grad_cI(x).transpose() * z);
        return linf_norm(KKT_0) / s_d;
    }
    template <typename Derived, typename Vec_x, typename Vec_cE, typename Vec_cI>
    inline double Eval_Complementary_Slackness_Infeasibility(
        objective<Derived> &f,
        const MatrixBase<Vec_x> &x,
        const MatrixBase<Vec_cE> &lbd,
        const MatrixBase<Vec_cI> &z,
        const double &mu,
        const double &s_max)
    {
        const int Nz = z.rows();
        const double s_c = std::max(s_max, (l1_norm(lbd) + l1_norm(z)) / (lbd.rows() + z.rows())) / s_max;
        spMat KKT_2 = f.Eval_cI(x).cwiseProduct(z) - dVec::Constant(Nz, mu);
        return linf_norm(KKT_2) / s_c;
    }

    template <typename Derived, typename Derived_B, typename Vec_x, typename Vec_cE, typename Vec_cI>
    inline double Eval_Barrier_Optimality_Error(objective<Derived> &f,
                                                barrier<Derived_B> &phi,
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

    template <typename Derived, typename Vec_x, typename Vec_cE, typename Vec_cI>
    inline double Eval_Global_Optimality_Error(objective<Derived> &f,
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