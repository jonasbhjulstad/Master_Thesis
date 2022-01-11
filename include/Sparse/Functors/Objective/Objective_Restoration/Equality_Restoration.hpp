#ifndef FIPOPT_EQUALITY_RESTORATION_HPP
#define FIPOPT_EQUALITY_RESTORATION_HPP

#include <Sparse/Functors/Objective/Objective.hpp>
#include <Sparse/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <Sparse/Functors/Objective/Objective_Messenger/Objective_Messenger.hpp>
#include <cmath>
#include <fstream>

namespace FIPOPT::Sparse
{

    template <typename Derived, template <class> typename Base = objective>
    struct equality_restoration_base : public Base<equality_restoration_base<Derived, Base>>
    {
        using base_t = Base<equality_restoration_base<Derived, Base>>;
        const int &Nx = base_t::Nx_;
        const int &Ng = base_t::Ng_;
        const int &Nh = base_t::Nh_;

    protected:
        dVec w_lb_;
        dVec w_ub_;
        spVec grad_f_;
        spMat hessian_h_;
        spMat grad_h_;
        objective<Derived> &f_;
        const dVec x_R_;
        dVec D_R_;
        spMat hessian_f_;
        const double zeta_;
        const double rho_;

    public:
        equality_restoration_base(objective<Derived> &f,
                                  const MatrixBase<dVec> &x_R,
                                  const double &mu,
                                  const double &rho = 1e3) : f_(f),
                                                             x_R_(x_R),
                                                             zeta_(sqrt(mu)),
                                                             rho_(rho)

        {

            base_t::Initialize(f.Get_Nx() + 2 * f.Get_Nh(), 0, f.Get_Nh());
            const int Nw = Nx + 2 * Nh;
            w_lb_.resize(Nw);
            w_lb_.topRows(Nx).setConstant(-1e20);
            dVec grad_f = dVec::Constant(Nx + 2 * Ng, rho);
            grad_f.topRows(Nx) = dVec::Constant(Nx, 0.);
            grad_f_ = grad_f.sparseView();
            std::vector<Triplet> T;
            for (int i = 0; i < Nh; i++)
            {
                T.push_back(Triplet(i, Nx + i, -1.));
                T.push_back(Triplet(i, Nx + Nh + i, 1.));
            }
            grad_h_.setFromTriplets(T.begin(), T.end());

            T.clear();
            spVec hessian_diagonal = zeta_ * x_R_.cwiseAbs().cwiseInverse().sparseView();
            set_diagonal(hessian_f_, hessian_diagonal);

            D_R_ = x_R_.cwiseInverse();
        }

        template <typename BaseType>
        inline Val operator()(const BaseType &w)
        {
            dVec x = w.head(Nx);
            dVec p = w.segment(Nx, Nh);
            dVec n = w.tail(Nh);
            return Val(rho_ * (p.sum() + n.sum()) + (zeta_ / 2) * (D_R_ * (x - x_R_)).norm());
        }

        template <typename BaseType>
        inline spVec Eval_grad(const BaseType &w)
        {
            dVec x = w.head(Nx);
            spVec g_f_x = (D_R_.cwiseProduct(x - x_R_)).cwiseAbs().sparseView();
            for (spVec::InnerIterator it(g_f_x); it; ++it)
            {
                grad_f_.insert(it.index()) = it.value();
            }
            return grad_f_;
        }

        template <typename BaseType>
        inline spMat Eval_hessian_f(const BaseType &w)
        {
            return hessian_f_;
        }

        template <typename BaseType>
        inline spVec Eval_h(const BaseType &w)
        {
            dVec x = w.topRows(Nx);
            dVec p = w.middleRows(Nx, Nh);
            dVec n = w.bottomRows(Nh);

            return f_.Eval_cE(x) - p + n;
        }

        template <typename BaseType>
        inline spMat Eval_grad_h(const BaseType &w)
        {

            spVec x = w.topRows(Nx);

            grad_h_.leftCols(Nx) = f_.Eval_grad_cE(x);

            return grad_h_;
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline spMat Eval_hessian_h(const BaseType_x &w, const BaseType_lbd &lbd)
        {
            spVec x = w.topRows(Nx);
            spMat hessian_cE = f_.Eval_hessian_cE(x, lbd);
            block_assign(hessian_h_, hessian_cE);
            return hessian_h_;
        }
        template <typename BaseType>
        inline spMat Eval_grad_g(const BaseType &x) { return spMat(); }
        template <typename BaseType>
        inline spVec Eval_g(const BaseType &x) { return spVec(); }

        inline spVec Get_x_lb()
        {
            return w_lb_.sparseView();
        }

        inline spVec Get_x_ub()
        {
            return w_ub_.sparseView();
        }
    };

    template <typename Derived>
    struct equality_restoration : public equality_restoration_base<Derived, objective>
    {
        using equality_restoration_base<Derived, objective>::equality_restoration_base;
    };

    template <typename Derived>
    equality_restoration(objective<Derived> &f,
                         const MatrixBase<dVec> &x_R,
                         const double &mu,
                         const double &rho) -> equality_restoration<Derived>;

    template <typename Derived>
    struct equality_restoration_memoized : public equality_restoration_base<Derived, objective_memoized>
    {
        using equality_restoration_base<Derived, objective_memoized>::equality_restoration_base;
    };

    template <typename Derived>
    equality_restoration_memoized(objective<Derived> &f,
                                  const MatrixBase<dVec> &x_R,
                                  const double &mu,
                                  const double &rho) -> equality_restoration_memoized<Derived>;

}

#endif