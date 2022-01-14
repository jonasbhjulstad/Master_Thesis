#ifndef FIPOPT_INITIAL_MULTIPLIERS_HPP
#define FIPOPT_INITIAL_MULTIPLIERS_HPP
#include <algorithm>
#include <Common/EigenDataTypes.hpp>
#include <Sparse/Functors/Objective/Objective.hpp>
#include <Dense/Initial_Multipliers/Initial_Multipliers.hpp>
#include <Eigen/SparseQR>

namespace FIPOPT::Sparse
{
    template <typename Vec>
    inline Eigen::Matrix<double, Vec::RowsAtCompileTime, 1> Eval_Bounded_Initial(const MatrixBase<Vec> &x0,
                                      const MatrixBase<Vec> &lb,
                                      const MatrixBase<Vec> &ub,
                                      const double &kappa_1,
                                      const double &kappa_2)
    {
        Vec lb_peturbed = lb + Vec((lb.array().abs().max(1) * (kappa_1)).min(kappa_2 * ((ub - lb).array())));
        Vec ub_peturbed = ub - Vec((ub.array().abs().max(1) * (kappa_1)).min(kappa_2 * ((ub - lb).array())));
        Vec x0_peturbed = x0.cwiseMax(lb_peturbed).cwiseMin(ub_peturbed);
        return x0_peturbed;
    }

    template <typename Derived, typename Vec_x>
    inline spMat Eval_Initial_Multipliers(
        objective<Derived> &f,
        const MatrixBase<Vec_x> &x0,
        const MatrixBase<Vec_x> &zl_0,
        const MatrixBase<Vec_x> &zu_0,
        const double &lbd_max = 100)
    {
        const int Nx = x0.rows();
        const int Ng = f.Get_Ng();
        const int Nh = f.Get_Nh();
        const int N_A = Nx + Nh;
        spMat grad_cE = f.Eval_grad_cE(x0);

        spMat A(N_A, N_A);
        set_diagonal(A, 1., 0, 0, Nx);

        spMat cE = f.Eval_grad_cE(x0);
        std::vector<Triplet> T;
        for (int k = 0; k < cE.outerSize(); ++k)
            for (spMat::InnerIterator it(cE, k); it; ++it)
            {
                T.push_back(Triplet(Nx + it.row(), it.col(), it.value()));
                T.push_back(Triplet(Nh + it.col(), it.row(), it.value()));
            }

        A.setFromTriplets(T.begin(), T.end());
        A.makeCompressed();

        spVec b = f.Eval_grad(x0) - zl_0 + zu_0;
        b.conservativeResize(N_A);
        Eigen::SparseQR<spMat, Eigen::COLAMDOrdering<int>> solver(A);
        spVec sol = solver.solve(-b);

        spVec lbd_0 = sol.bottomRows(Nh);

        for (spVec::InnerIterator it(lbd_0, 0); it; ++it)
        {
            if (it.value() > lbd_max)
                lbd_0.coeffRef(it.index(),0) = lbd_max;
        }

        return lbd_0;
    }

    template <typename Derived>
    inline spMat Eval_Initial_Multipliers(
        objective<Derived> &f,
        const SparseMatrixBase<spVec> &x0,
        const double &lbd_max = 100)
        {
            const int Nx = x0.rows();
            spVec zl_0 = dVec::Constant(Nx, 0.).sparseView();
            spVec zu_0 = dVec::Constant(Nx, 1e20).sparseView();
            return Eval_Initial_Multipliers(f, x0, zl_0, zu_0);
        }

}

#endif