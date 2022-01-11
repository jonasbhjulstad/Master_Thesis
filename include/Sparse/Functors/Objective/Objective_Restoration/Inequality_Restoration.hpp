#ifndef FIPOPT_INEQUALITY_RESTORATION_HPP
#define FIPOPT_INEQUALITY_RESTORATION_HPP

#include <Sparse/Functors/Objective/Objective.hpp>
#include <Sparse/Functors/Objective/Objective_Memoized/Objective_Memoized.hpp>
#include <cmath>
#include <fstream>

namespace FIPOPT::Sparse
{

    template <typename Derived, template <class> typename Base = objective>
    struct inequality_restoration_base : public Base<inequality_restoration_base<Derived, Base>>
    {
        using base_t = Base<inequality_restoration_base<Derived, Base>>;
        const int& Nx = base_t::Nx_;
        const int& Ng = base_t::Ng_;

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
                                                                     Delta_cI_(std::min(kappa_cI * f.Eval_cI(x_R).topRows(Ng).norm(), 1e3))

        {
            base_t::Initialize(f.Get_Nx() + 2 * f.Get_Ng(), 0, f.Get_Ng());
            const int Nw = Nx + 2 * Ng;
            w_lb_.resize(Nw);
            w_lb_.topRows(Nx).setConstant(-1e20);
            dVec grad_f = dVec::Constant(Nx + 2*Ng, rho);
            grad_f.head(Nx) = dVec::Constant(Nx, 0.);
            grad_f_ = grad_f.sparseView();
            std::vector<Triplet> Tgf;
            std::vector<Triplet> T;
            for (int i = 0; i < Ng; i++)
            {
                T.push_back(Triplet(i, Nx + i, -1.));
                T.push_back(Triplet(i, Nx + Ng + i, 1.));
            }
            grad_h_.setFromTriplets(T.begin(), T.end());

            T.clear();
            spVec hessian_diagonal = zeta_*x_R_.cwiseAbs().cwiseInverse().sparseView();
            set_diagonal(hessian_f_, hessian_diagonal);

            D_R_ = x_R_.cwiseInverse();
        }

        template <typename BaseType>
        inline Val operator()(const BaseType &w)
        {
            dVec x = w.head(Nx);
            dVec p = w.segment(Nx, Ng);
            dVec n = w.tail(Ng);
            return Val(rho_ * (p.sum() + n.sum()) + (zeta_ / 2) * (D_R_.cwiseProduct(x - x_R_)).norm());
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
            dVec x = w.head(Nx);
            dVec p = w.segment(Nx, Ng);
            dVec n = w.tail(Ng);
            return dVec(f_.Eval_cI(x).topRows(Ng) - p + n - dVec::Constant(Ng, Delta_cI_)).sparseView(); // - spVec::Constant(Delta_cI_);
        }

        template <typename BaseType>
        inline spMat Eval_grad_h(const BaseType &w)
        {

            spVec x = w.topRows(Nx);

            grad_h_.leftCols(Nx) = f_.Eval_grad_cI(x).topRows(Ng);

            return grad_h_;
        }

        template <typename BaseType_x, typename BaseType_lbd>
        inline spMat Eval_hessian_h(const BaseType_x &w, const BaseType_lbd &lbd)
        {
            spVec x = w.topRows(Nx);
            spMat hessian_cI = f_.Eval_hessian_cI(x, lbd);
            block_assign(hessian_h_, hessian_cI);

            return hessian_h_;
        }

        template <typename BaseType>
        inline spMat Eval_grad_g(const BaseType& x){return spMat();}
        template <typename BaseType>
        inline spVec Eval_g(const BaseType& x){return spVec();}


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
    struct inequality_restoration : public inequality_restoration_base<Derived, objective>
    {
        using inequality_restoration_base<Derived, objective>::inequality_restoration_base;
    };

    template <typename Derived>
    inequality_restoration(objective<Derived> &f,
                           const MatrixBase<dVec> &x_R,
                           const double &mu,
                           const double &rho) -> inequality_restoration<Derived>;

    template <typename Derived>
    struct inequality_restoration_memoized : public inequality_restoration_base<Derived, objective_memoized>
    {
        using inequality_restoration_base<Derived, objective_memoized>::inequality_restoration_base;
    };

    template <typename Derived>
    inequality_restoration_memoized(objective<Derived> &f,
                                    const MatrixBase<dVec> &x_R,
                                    const double &mu,
                                    const double &rho) -> inequality_restoration_memoized<Derived>;
}

#endif