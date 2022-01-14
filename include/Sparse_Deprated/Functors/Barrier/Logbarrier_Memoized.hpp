#ifndef FIPOPT_LOGBARRIER_MEMOIZED_Sparse_HPP
#define FIPOPT_LOGBARRIER_MEMOIZED_Sparse_HPP
#include <Sparse/Functors/Objective/Objective.hpp>
#include <Sparse/Functors/Barrier/Barrier_Memoized/Barrier_Memoized.hpp>
#include <Sparse/Functors/Barrier/Logbarrier.hpp>

namespace FIPOPT::Sparse
{

    template <typename Derived>
    struct logbarrier_memoized : public logbarrier_base<Derived, barrier_memoized>
    {
        using logbarrier_base<Derived, barrier_memoized>::logbarrier_base;
    };

    template <typename Derived>
    logbarrier_memoized(objective<Derived>& f, const double& mu) -> logbarrier_memoized<Derived>;
}

#endif