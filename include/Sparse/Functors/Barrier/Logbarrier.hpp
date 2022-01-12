#ifndef FIPOPT_LOGBARRIER_DENSE_HPP
#define FIPOPT_LOGBARRIER_DENSE_HPP
#include <Common/EigenDataTypes.hpp>
#include <Dense/Functors/Barrier/Barrier.hpp>
#include <Dense/Functors/Objective/Objective.hpp>
#include <Dense/Functors/Barrier/Barrier_Messenger/Barrier_Messenger.hpp>
#include <Dense/Functors/Barrier/Barrier_Journalist/Barrier_Journalist.hpp>

namespace FIPOPT::Sparse
{

    template <typename Derived, template <class,int,int,int> typename Base = barrier>
    struct logbarrier_base : public Base<logbarrier_base<Derived, Base>>
    {
        using Objective = objective<Derived>;
        using base_t = Base<logbarrier_base<Derived, Base>>;

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
            dVec barrier_term = dVec(f_.Eval_cI(x).array().log());
            barrier_term = dVec((barrier_term.array().isInf()).select(0., barrier_term.array()));
            return f_(x) - Val(mu_ * (barrier_term.array().sum()));
        }

        template <typename BaseType>
        inline dVec Eval_grad(const BaseType &x)
        {
            dVec s_inv = f_.Eval_cI(x).cwiseInverse();
            s_inv = dVec((s_inv.array() > 0).select(s_inv, dVec::Zero()));
            return f_.Eval_grad(x) - mu_ * f_.Eval_grad_cI(x).transpose() *(s_inv);
        }

        template <typename BaseType>
        inline dVec Eval_cE(const BaseType &x)
        {
            return f_.Eval_cE(x);
        }

        template <typename BaseType>
        inline dVec Eval_cI(const BaseType &x)
        {
            return f_.Eval_cI(x);
        }

        inline double Get_mu()
        {
            return mu_;
        }
    };

    template <typename Derived>
    struct logbarrier : public logbarrier_base<Derived, barrier>
    {
        using logbarrier_base<Derived, barrier>::logbarrier_base;
    };

    template <typename Derived>
    logbarrier(objective<Derived>& f, const double& mu) -> logbarrier<Derived>;


}

#endif