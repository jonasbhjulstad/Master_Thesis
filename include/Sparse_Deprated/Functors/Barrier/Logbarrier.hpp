#ifndef FIPOPT_LOGBARRIER_Sparse_HPP
#define FIPOPT_LOGBARRIER_Sparse_HPP
#include <Common/EigenDataTypes.hpp>
#include <Sparse/Functors/Barrier/Barrier.hpp>
#include <Sparse/Functors/Objective/Objective.hpp>
#include <Sparse/Functors/Barrier/Barrier_Messenger/Barrier_Messenger.hpp>
#include <Sparse/Functors/Barrier/Barrier_Journalist/Barrier_Journalist.hpp>

namespace FIPOPT::Sparse
{

    template <typename Derived, template <class> typename Base = barrier>
    struct logbarrier_base : public Base<logbarrier_base<Derived, Base>>
    {
        using Objective = objective<Derived>;

        using base_t = Base<logbarrier_base<Derived, Base>>;

    protected:
        objective<Derived> &f_;
        const double mu_;

    public:
        template <typename Observer>
        logbarrier_base(objective<Derived> &f, const double &mu, Observer &obs) : f_(f), mu_(mu), base_t(obs, mu) {}

        logbarrier_base(objective<Derived> &f, const double &mu) : f_(f), mu_(mu) {}

        template <typename BaseType>
        inline spVal operator()(const BaseType &x)
        {
            spVal res = f_(x).sparseView();
            spVec cI = f_.Eval_cI(x);
            for (spVec::InnerIterator it(cI); it; ++it)
            {
                res.coeffRef(0) += mu_ * log(it.value());
            }
            return res;
        }

        template <typename BaseType>
        inline spVec Eval_grad(const BaseType &x)
        {
            spVec s_inv = f_.Eval_cI(x).cwiseInverse();
            return f_.Eval_grad(x) - mu_ * f_.Eval_grad_cI(x).transpose() * (s_inv);
        }

        template <typename BaseType>
        inline spVec Eval_cE(const BaseType &x)
        {
            return f_.Eval_cE(x);
        }

        template <typename BaseType>
        inline spVec Eval_cI(const BaseType &x)
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
    logbarrier(objective<Derived> &f, const double &mu) -> logbarrier<Derived>;

    // template <typename Derived>
    // struct logbarrier_messenger : public logbarrier_base<Derived, barrier_messenger>
    // {
    //     using logbarrier_base<Derived, barrier_journalist>::logbarrier_base;
    // };

}

#endif