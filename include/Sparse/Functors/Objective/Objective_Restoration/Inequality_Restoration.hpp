#ifndef FIPOPT_INEQUALITY_RESTORATION_HPP
#define FIPOPT_INEQUALITY_RESTORATION_HPP

#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <cmath>
#include <fstream>

namespace FIPOPT::Sparse
{

    template <typename Derived, template <class, int, int, int> typename Base = objective>
    struct inequality_restoration_base : public Base<inequality_restoration_base<Derived, Base>, Nx + 2 * Ng, 0, Ng>
    {
        constexpr static int Nw = Nx + 2 * Ng;
        using Objective = objective<Derived>;
        using Mat_w = Eigen::Matrix<double, Nw, Nw>;
        using Vec_w = Eigen::Matrix<double, Nw, 1>;
        using spMat = Eigen::Matrix<double, Nx, Nx>;
        using dVec = Eigen::Matrix<double, Nx, 1>;
        using spMat = Eigen::Matrix<double, Ng, Nw>;
        using dVec = Eigen::Matrix<double, Ng, 1>;
        using base_t = Base<inequality_restoration_base<Derived, Base>, Nx + 2 * Ng, 0, Ng>;

    protected:
        Vec_w w_lb_ = Vec_w::Constant(0.);
        Vec_w w_ub_ = Vec_w::Constant(1e20);
        Vec_w grad_f_;
        Mat_w hessian_h_ = Mat_w::Zero();
        spMat grad_h_ = spMat::Zero();
        Objective &f_;
        const dVec x_R_;
        spMat D_R_ = spMat::Zero();
        Mat_w hessian_f_ = Mat_w::Zero();
        const double zeta_;
        const double rho_;
        const double Delta_cI_;

    public:
        inequality_restoration_base(objective<Derived> &f,
                                         const MatrixBase<dVec> &x_R,
                                         const double &mu,
                                         const double &rho = 1,
                                         const double &kappa_cI = 1e-4) : f_(f),
                                                                          x_R_(x_R),
                                                                          zeta_(sqrt(mu)),
                                                                          rho_(rho),
                                                                          Delta_cI_(std::min(kappa_cI * f.Eval_cI(x_R).head(Ng).norm(), 1e3))

        {
            w_lb_.head(Nx) = dVec::Constant(-1e20);
            grad_f_.tail(2 * Ng).setConstant(rho);
            grad_h_.middleCols(Nx, Ng).diagonal().setConstant(-1.);
            grad_h_.rightCols(Ng).setIdentity();
            D_R_.diagonal() = x_R_.cwiseInverse();
            hessian_f_.topLeftCorner(Nx, Nx).diagonal() = zeta_*D_R_.diagonal();
        }

        template <typename BaseType>
        inline Val operator()(const BaseType &w)
        {
            dVec x = w.head(Nx);
            dVec p = w.segment(Nx, Ng);
            dVec n = w.tail(Ng);
            return Val(rho_ * (p.sum() + n.sum()) + (zeta_ / 2) * (D_R_ * (x - x_R_)).template lpNorm<2>());
        }

        template <typename BaseType>
        inline Vec_w Eval_grad(const BaseType &w)
        {
            dVec x = w.head(Nx);
            grad_f_.head(Nx) = zeta_*(D_R_ * (x - x_R_)).cwiseAbs();
            return grad_f_;
        }

        template <typename BaseType>
        inline Mat_w Eval_hessian_f(const BaseType &w)
        {
            return hessian_f_;
        }

        template <typename BaseType>
        inline dVec Eval_h(const BaseType &w)
        {
            dVec x = w.head(Nx);
            dVec p = w.segment(Nx, Ng);
            dVec n = w.tail(Ng);

            return f_.Eval_cI(x).head(Ng) - p + n - dVec::Constant(Delta_cI_);// - dVec::Constant(Delta_cI_);
        }

        template <typename BaseType>
        inline spMat Eval_grad_h(const BaseType &w)
        {

            dVec x = w.head(Nx);

            grad_h_.leftCols(Nx) = f_.Eval_grad_cI(x).topRows(Ng);

            return grad_h_;
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline Mat_w Eval_hessian_h(const BaseType_x &w, const BaseType_lbd &lbd)
        {
            dVec x = w.head(Nx);
            hessian_h_.topLeftCorner(Nx, Nx) = f_.Eval_hessian_cI(x, lbd);
            return hessian_h_;
        }


        inline Vec_w Get_x_lb()
        {
            return w_lb_;
        }

        inline Vec_w Get_x_ub()
        {
            return w_ub_;
        }
    };

    template <typename Derived>
    struct inequality_restoration : public inequality_restoration_base<Derived, objective>
    {
        using inequality_restoration_base<Derived, objective>::inequality_restoration_base;
    };

    template <typename Derived>
    inequality_restoration(objective<Derived> &f,
                                const MatrixBase<Eigen::Matrix<double, Nx, 1>> &x_R,
                                const double &mu,
                                const double &rho) -> inequality_restoration<Derived>;

    template <typename Derived>
    struct inequality_restoration_memoized : public inequality_restoration_base<Derived, objective_memoized>
    {
        using inequality_restoration_base<Derived, objective_memoized>::inequality_restoration_base;
    };

    template <typename Derived>
    inequality_restoration_memoized(objective<Derived> &f,
                                         const MatrixBase<Eigen::Matrix<double, Nx, 1>> &x_R,
                                         const double &mu,
                                         const double &rho) -> inequality_restoration_memoized<Derived>;
}

#endif