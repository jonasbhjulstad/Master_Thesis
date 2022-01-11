#ifndef FIPOPT_EQUALITY_RESTORATION_HPP
#define FIPOPT_EQUALITY_RESTORATION_HPP

#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <Dense/Functors/Objective/Objective_Messenger/Objective_Messenger.hpp>
#include <cmath>
#include <fstream>

namespace FIPOPT::Dense
{

    template <typename Derived, int Nx, int Ng, int Nh, template <class, int, int, int> typename Base = objective>
    struct equality_restoration_base : public Base<equality_restoration_base<Derived, Nx, Ng, Nh, Base>, Nx + 2 * Nh, 0, Nh>
    {
        constexpr static int Nw = Nx + 2 * Nh;
        using Objective = objective<Derived, Nx, Ng, Nh>;
        using Mat_w = Eigen::Matrix<double, Nw, Nw>;
        using Vec_w = Eigen::Matrix<double, Nw, 1>;
        using Mat_x = Eigen::Matrix<double, Nx, Nx>;
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Mat_h = Eigen::Matrix<double, Nh, Nw>;
        using Vec_h = Eigen::Matrix<double, Nh, 1>;
        using base_t = Base<equality_restoration_base<Derived, Nx, Ng, Nh, Base>, Nx + 2 * Nh, 0, Nh>;

    protected:
        Vec_w w_lb_ = Vec_w::Constant(0.);
        Vec_w w_ub_ = Vec_w::Constant(1e20);
        Vec_w grad_f_;
        Mat_w hessian_h_ = Mat_w::Zero();
        Mat_h grad_h_ = Mat_h::Zero();
        Objective &f_;
        const Vec_x x_R_;
        Mat_x D_R_ = Mat_x::Zero();
        Mat_w hessian_f_ = Mat_w::Zero();
        const double zeta_;
        const double rho_;

    public:
        equality_restoration_base(objective<Derived, Nx, Ng, Nh> &f,
                                        const MatrixBase<Vec_x> &x_R,
                                        const double &mu,
                                        const double &rho = 1e3) : f_(f),
                                                                   x_R_(x_R),
                                                                   zeta_(sqrt(mu)),
                                                                   rho_(rho)

        {
            w_lb_.head(Nx) = Vec_x::Constant(-1e20);
            grad_f_.tail(2 * Nh).setConstant(rho);
            grad_h_.middleCols(Nx, Nh).diagonal().setConstant(-1.);
            grad_h_.rightCols(Nh).setIdentity();
            D_R_.diagonal() = x_R_.cwiseInverse();
            hessian_f_.topLeftCorner(Nx, Nx).diagonal() = zeta_*D_R_.diagonal();
        }

        template <typename BaseType>
        inline Val operator()(const BaseType &w)
        {
            Vec_x x = w.head(Nx);
            Vec_h p = w.segment(Nx, Nh);
            Vec_h n = w.tail(Nh);
            return Val(rho_ * (p.sum() + n.sum()) + (zeta_ / 2) * (D_R_ * (x - x_R_)).template lpNorm<2>());
        }

        template <typename BaseType>
        inline Vec_w Eval_grad(const BaseType &w)
        {
            Vec_x x = w.head(Nx);
            grad_f_.head(Nx) = zeta_*(D_R_ * (x - x_R_)).cwiseAbs();
            return grad_f_;
        }

        template <typename BaseType>
        inline Mat_w Eval_hessian_f(const BaseType &w)
        {
            return hessian_f_;
        }

        template <typename BaseType>
        inline Vec_h Eval_h(const BaseType &w)
        {
            Vec_x x = w.head(Nx);
            Vec_h p = w.segment(Nx, Nh);
            Vec_h n = w.tail(Nh);

            return f_.Eval_cE(x) - p + n;
        }

        template <typename BaseType>
        inline Mat_h Eval_grad_h(const BaseType &w)
        {

            Vec_x x = w.head(Nx);

            grad_h_.leftCols(Nx) = f_.Eval_grad_cE(x);

            return grad_h_;
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline Mat_w Eval_hessian_h(const BaseType_x &w, const BaseType_lbd &lbd)
        {
            Vec_x x = w.head(Nx);
            hessian_h_.topLeftCorner(Nx, Nx) = f_.Eval_hessian_cE(x, lbd);
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

    template <typename Derived, int Nx, int Ng, int Nh>
    struct equality_restoration : public equality_restoration_base<Derived, Nx, Ng, Nh, objective>
    {
        using equality_restoration_base<Derived, Nx, Ng, Nh, objective>::equality_restoration_base;
    };

    template <typename Derived, int Nx, int Ng, int Nh>
    equality_restoration(objective<Derived, Nx, Ng, Nh> &f,
                               const MatrixBase<Eigen::Matrix<double, Nx, 1>> &x_R,
                               const double &mu,
                               const double &rho) -> equality_restoration<Derived, Nx, Ng, Nh>;

    template <typename Derived, int Nx, int Ng, int Nh>
    struct equality_restoration_memoized : public equality_restoration_base<Derived, Nx, Ng, Nh, objective_memoized>
    {
        using equality_restoration_base<Derived, Nx, Ng, Nh, objective_memoized>::equality_restoration_base;
    };

    template <typename Derived, int Nx, int Ng, int Nh>
    equality_restoration_memoized(objective<Derived, Nx, Ng, Nh> &f,
                                        const MatrixBase<Eigen::Matrix<double, Nx, 1>> &x_R,
                                        const double &mu,
                                        const double &rho) -> equality_restoration_memoized<Derived, Nx, Ng, Nh>;

}

#endif