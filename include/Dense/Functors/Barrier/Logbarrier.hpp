#ifndef FIPOPT_LOGBARRIER_DENSE_HPP
#define FIPOPT_LOGBARRIER_DENSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Barrier/Barrier.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Barrier/Barrier_Messenger/Barrier_Messenger.hpp>
#include <Dense/Functors/Barrier/Barrier_Journalist/Barrier_Journalist.hpp>

namespace FIPOPT::Dense
{

    template <typename Derived, int Nx, int Ng, int Nh, template <class,int,int,int> typename Base = barrier>
    struct logbarrier_base : public Base<logbarrier_base<Derived, Nx, Ng, Nh, Base>, Nx, Ng, Nh>
    {
        using Objective = objective<Derived, Nx, Ng, Nh>;
        // Dense types
        using Vec_x = Eigen::Matrix<double, Nx, 1>;
        using Vec_cE = Eigen::Matrix<double, Nh, 1>;
        using Vec_cI = Eigen::Matrix<double, Ng + 2*Nx, 1>;
        using base_t = Base<logbarrier_base<Derived, Nx, Ng, Nh, Base>, Nx, Ng, Nh>;

    protected:
        Objective &f_;
        const double mu_;

    public:
        template <typename Observer>
        logbarrier_base(Objective &f, const double &mu, Observer& obs) : f_(f), mu_(mu), base_t(obs, mu) {}

        logbarrier_base(Objective &f, const double &mu) : f_(f), mu_(mu) {}

        template <typename BaseType>
        inline Val operator()(const BaseType &x)
        {
            Vec_cI barrier_term = Vec_cI(f_.Eval_cI(x).array().log());
            barrier_term = Vec_cI((barrier_term.array().isInf()).select(0., barrier_term.array()));
            return f_(x) - Val(mu_ * (barrier_term.array().sum()));
        }

        template <typename BaseType>
        inline Vec_x Eval_grad(const BaseType &x)
        {
            Vec_cI s_inv = f_.Eval_cI(x).cwiseInverse();
            s_inv = Vec_cI((s_inv.array() > 0).select(s_inv, Vec_cI::Zero()));
            return f_.Eval_grad(x) - mu_ * f_.Eval_grad_cI(x).transpose() *(s_inv);
        }

        template <typename BaseType>
        inline Vec_cE Eval_cE(const BaseType &x)
        {
            return f_.Eval_cE(x);
        }

        template <typename BaseType>
        inline Vec_cI Eval_cI(const BaseType &x)
        {
            return f_.Eval_cI(x);
        }

        inline double Get_mu()
        {
            return mu_;
        }
    };

    template <typename Derived, int Nx, int Ng, int Nh>
    struct logbarrier : public logbarrier_base<Derived, Nx, Ng, Nh, barrier>
    {
        using logbarrier_base<Derived, Nx, Ng, Nh, barrier>::logbarrier_base;
    };

    template <typename Derived, int Nx, int Ng, int Nh>
    logbarrier(objective<Derived, Nx, Ng, Nh>& f, const double& mu) -> logbarrier<Derived, Nx, Ng, Nh>;


}

#endif