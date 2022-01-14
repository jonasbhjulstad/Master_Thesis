#ifndef FIPOPT_KKT_SYSTEM_HPP
#define FIPOPT_KKT_SYSTEM_HPP

#include <Sparse/Functors/Objective/Objective.hpp>
#include <Sparse/Functors/Barrier/Barrier.hpp>
#include <Common/EigenDataTypes.hpp>
#include <Sparse/Optimality/Optimality.hpp>
#include <Common/Utils/Eigen_Utils.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <type_traits>
#include <fstream>
namespace FIPOPT::Sparse
{
    template <typename Derived, typename Vec_x, typename Vec_cE, typename Vec_cI>
    inline void Eval_KKT_Jacobian(
        Derived &f,
        spMat &KKT_mat,
        const SparseMatrixBase<Vec_x> &x,
        const SparseMatrixBase<Vec_cE> &lbd,
        const SparseMatrixBase<Vec_cI> &z,
        const double &mu)
    {
        const int Nx = x.rows();
        const int Nh = lbd.rows();
        spMat Sigma = f.Eval_cI(x).cwiseInverse().cwiseProduct(z);

        spMat PD_lagrangian = f.Eval_hessian(x, lbd) + (f.Eval_grad_cI(x).transpose() * Sigma).cwiseProduct(f.Eval_grad_cI(x));
        //[..       grad_c]
        //[grad_c^T    ..]

        std::vector<Triplet> T = to_triplets(PD_lagrangian);

        spMat grad_cE = f.Eval_grad_cE(x);

        for (int k=0; k<grad_cE.outerSize(); ++k)
            for (SparseMatrix<double>::InnerIterator it(grad_cE,k); it; ++it)
            {
                    T.push_back(Triplet(Nx + it.row(), it.col(), it.value()));
                    T.push_back(Triplet(it.col(), Nx + it.row(), it.value()));
            }
        
        KKT_mat.setFromTriplets(T.begin(), T.end());

    }

    template <typename Derived, typename Derived_B, typename Vec_x, typename Vec_cE, typename Vec_cI>
    inline void Eval_KKT_Value(
        objective<Derived> &f,
        barrier<Derived_B> &phi,
        spVec &KKT_vec,
        const SparseMatrixBase<Vec_x> &x,
        const SparseMatrixBase<Vec_cE> &lbd,
        const SparseMatrixBase<Vec_cI> &z)
    {
        const int Nx = x.rows();
        const int Nh = lbd.rows();
        spVec top = -(phi.Eval_grad(x) - f.Eval_grad_cE(x).transpose() * lbd);
        spVec bottom = -f.Eval_cE(x);


        for (spVec::InnerIterator it(top); it; ++it)
        {
            KKT_vec.insert(it.index()) = it.value();
        }
        
        for (spVec::InnerIterator it(bottom); it; ++it)
        {
            KKT_vec.insert(Nx + it.index()) = it.value();
        }

    }

    template <typename Mat_A, typename Vec_x, typename Vec_cE, typename SolverBase>
    inline bool Solve_KKT_System(const SparseMatrixBase<Mat_A> &KKT_vec,
                                 SparseMatrixBase<Vec_x> &d_x,
                                 SparseMatrixBase<Vec_cE> &d_lbd,
                                 SolverBase &KKT_solver)
    {
        const int Nx = d_x.rows();
        const int Nh = d_lbd.rows();

        spMat sol = KKT_solver.solve(KKT_vec);
        d_x = sol.topRows(Nx);
        d_lbd = -sol.bottomRows(Nh);
        return !any_nan(sol);
    }

    template <typename Vec_x, typename Vec_cE, typename Vec_cI>
    inline void Update_PD_States(
        SparseMatrixBase<Vec_x> &x,
        SparseMatrixBase<Vec_cE> &lbd,
        SparseMatrixBase<Vec_cI> &z,
        SparseMatrixBase<Vec_x> &d_x,
        SparseMatrixBase<Vec_cE> &d_lbd,
        SparseMatrixBase<Vec_cI> &d_z,
        const double &alpha,
        const double alpha_z_g,
        const double alpha_z_ub,
        const double alpha_z_lb)
    {
        const int Nx = x.rows();
        const int Nh = lbd.rows();
        const int Ng = z.rows() - 2*Nx;
        x += alpha*d_x;
        lbd += alpha * d_lbd;

        dVec alpha_d_z(Ng + 2*Nx);
        alpha_d_z.head(Ng) = alpha_z_g*d_z.topRows(Ng);
        alpha_d_z.segment(Ng, Nx) = alpha_z_ub*d_z.middleRows(Ng, Nx);
        alpha_d_z.tail(Nx) = alpha_z_lb*d_z.bottomRows(Nx);

        z += alpha_d_z.sparseView();
    }

}
#endif